from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.dataset.parquet_loader import load_first_clean_play_ticks
from cs2_ai.ml.models.aim_attention import AimAttentionModel
from cs2_ai.ml.models.decision_dqn import DecisionDQN
from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
from cs2_ai.ml.utils.torch_utils import get_device, torch_available
from cs2_ai.pipeline.offline_ai_pipeline import OfflineAIPipeline
from cs2_ai.state.game_state_builder import GameStateBuilder


def run_torch_smoke_test() -> None:
    if not torch_available():
        print("PyTorch unavailable. Install torch to run ML model smoke-test.")
        return
    import torch

    device = get_device()
    print("PyTorch available")
    print(f"Device: {device}")

    batch = 2
    seq_len = 8
    input_dim = 32
    action_dim = 6
    dummy = torch.randn(batch, seq_len, input_dim, device=device)

    enemy_model = EnemyTrackerLSTM(input_dim=input_dim).to(device)
    positions, confidence = enemy_model(dummy)
    assert tuple(positions.shape) == (2, 5, 3)
    assert tuple(confidence.shape) == (2, 5)
    print("EnemyTrackerLSTM dummy forward OK")

    aim_model = AimAttentionModel(input_dim=input_dim).to(device)
    aim_delta, shoot_logits, rightclick_logits = aim_model(dummy)
    assert tuple(aim_delta.shape) == (2, 2)
    assert tuple(shoot_logits.shape) == (2, 1)
    assert tuple(rightclick_logits.shape) == (2, 1)
    print("AimAttentionModel dummy forward OK")

    decision_model = DecisionDQN(input_dim=input_dim, action_dim=action_dim).to(device)
    q_values = decision_model(dummy)
    assert tuple(q_values.shape) == (2, action_dim)
    print("DecisionDQN dummy forward OK")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    try:
        parquet_path, df = load_first_clean_play_ticks(dataset_dir)
    except FileNotFoundError:
        print("No clean_play_ticks parquet found. Run parser/cleaner first.")
        run_torch_smoke_test()
        return 0

    alive_ticks = df[df["is_alive"] == True] if "is_alive" in df.columns else df
    if alive_ticks.empty:
        print("No alive players found in clean_play_ticks dataset.")
        run_torch_smoke_test()
        return 0

    target_tick = int(alive_ticks.iloc[0]["tick"])
    tick_rows = df[df["tick"] == target_tick].copy()
    perspective_steamid = int(tick_rows[tick_rows["is_alive"] == True].iloc[0]["steamid"])

    builder = GameStateBuilder()
    game_state = builder.build_from_tick_rows(tick_rows, perspective_steamid)
    pipeline = OfflineAIPipeline(memory_len=8)
    action_plan = pipeline.step(game_state)

    print(f"Dataset: {parquet_path.name}")
    print(f"Tick: {target_tick}")
    print("GameState summary:")
    pprint({
        "perspective_steamid": game_state.perspective_steamid,
        "self_name": game_state.self_player.name,
        "team_num": game_state.self_player.team_num,
        "weapon": game_state.self_player.weapon,
        "hp": game_state.self_player.health,
        "teammates": [player.name for player in game_state.teammates],
        "enemies": [player.name for player in game_state.enemies],
    })
    print("EnemyTrackerOutput:")
    pprint(asdict(pipeline.last_enemy_tracker_output))
    print("BeliefState:")
    pprint(asdict(pipeline.last_belief_state))
    print("DecisionOutput:")
    pprint(asdict(pipeline.last_decision_output))
    print("MovementOutput:")
    pprint(asdict(pipeline.last_movement_output))
    print("AimShootOutput:")
    pprint(asdict(pipeline.last_aim_output))
    print("BuyOutput:")
    pprint(asdict(pipeline.last_buy_output))
    print("ActionPlan:")
    pprint(asdict(action_plan))
    run_torch_smoke_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
