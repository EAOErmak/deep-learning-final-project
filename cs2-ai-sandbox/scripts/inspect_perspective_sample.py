from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cs2_ai.dataset.parquet_loader import load_first_clean_play_ticks
from cs2_ai.dataset.perspective_builder import PerspectiveSampleBuilder
from cs2_ai.state.game_state_builder import GameStateBuilder

try:
    from rich import print as rprint
except Exception:
    rprint = print


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    try:
        parquet_path, df = load_first_clean_play_ticks(dataset_dir)
    except FileNotFoundError:
        print("No clean_play_ticks parquet found. Run parser/cleaner first.")
        return 0

    alive_df = df[df["is_alive"] == True] if "is_alive" in df.columns else df
    if alive_df.empty:
        print("No alive players found in clean_play_ticks dataset.")
        return 0

    random_tick = int(random.choice(alive_df["tick"].unique().tolist()))
    tick_rows = df[df["tick"] == random_tick].copy()
    sample_builder = PerspectiveSampleBuilder(GameStateBuilder())
    samples = sample_builder.build_samples_for_tick(tick_rows, alive_only=True)
    if not samples:
        print("No perspective samples could be built for selected tick.")
        return 0

    rprint(f"Dataset: {parquet_path.name}")
    rprint(f"Tick: {random_tick}")
    for sample in samples[:3]:
        state = sample.game_state
        payload = {
            "perspective_name": state.self_player.name,
            "perspective_steamid": state.self_player.steamid,
            "self_team": state.self_player.team_num,
            "teammates": [player.name for player in state.teammates],
            "enemies": [player.name for player in state.enemies],
            "target_input": asdict(sample.target_input),
            "self_weapon": state.self_player.weapon,
            "self_hp": state.self_player.health,
            "self_position": state.self_player.position,
        }
        rprint(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
