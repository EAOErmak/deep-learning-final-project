from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cs2_ai.ml.models.decision_dqn import DecisionDQN
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available
from game_state import GameState

ActionDict = dict[str, Any]


class NeuralRuntimeAgent:
    FEATURE_ORDER = [
        'self_x',
        'self_y',
        'self_z',
        'self_vel_x',
        'self_vel_y',
        'self_vel_z',
        'self_hp',
        'self_armor',
        'self_money',
        'self_alive',
        'ammo',
        'ammo_reserve',
        'yaw',
        'pitch',
        'team_is_ct',
        'round_live',
        'round_freeze',
        'round_warmup',
        'bomb_planted',
        'visible_players_count',
        'available_players_count',
        'has_spatial_state',
        'has_enemy_context',
        'enemy_visible',
        'enemy_rel_x',
        'enemy_rel_y',
        'enemy_rel_z',
        'enemy_hp',
        'enemy_distance',
    ]

    def __init__(self, seed: int = 42, mouse_scale: float = 18.0, checkpoint_path: str | None = None) -> None:
        if not torch_available():
            raise RuntimeError('PyTorch is not available. Install torch to use NeuralRuntimeAgent.')

        import torch

        set_seed(seed)
        self.torch = torch
        self.device = get_device()
        self.mouse_scale = mouse_scale
        self.logger = logging.getLogger(__name__)
        self.model = DecisionDQN(input_dim=len(self.FEATURE_ORDER), action_dim=10).to(self.device)
        self.model.eval()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            self.logger.info('NeuralRuntimeAgent initialized | device=%s | checkpoint=%s | input_dim=%s', self.device, checkpoint_path, len(self.FEATURE_ORDER))
        else:
            self.logger.info('NeuralRuntimeAgent initialized | device=%s | seed=%s | input_dim=%s', self.device, seed, len(self.FEATURE_ORDER))

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_file}')
        checkpoint = self.torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f'Unsupported checkpoint format: {checkpoint_file}')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, features: dict[str, float | int | bool]) -> ActionDict:
        return self._predict_from_features(features)

    def predict_state(
        self,
        _game_state: GameState,
        features: dict[str, float | int | bool] | None = None,
    ) -> ActionDict:
        return self._predict_from_features(features or {})

    def _predict_from_features(self, features: dict[str, float | int | bool]) -> ActionDict:
        vector = [float(features.get(name, 0.0)) for name in self.FEATURE_ORDER]
        x = self.torch.tensor(vector, dtype=self.torch.float32, device=self.device).unsqueeze(0)

        with self.torch.no_grad():
            logits = self.model(x).squeeze(0)

        move_logits = logits[:4]
        aux_logits = logits[4:8]
        mouse_logits = logits[8:10]

        move_probs = self.torch.softmax(move_logits, dim=0)
        move_idx = int(self.torch.multinomial(move_probs, num_samples=1).item())
        aux_probs = self.torch.sigmoid(aux_logits)
        aux_samples = self.torch.bernoulli(aux_probs).to(dtype=self.torch.int32)
        mouse_values = self.torch.tanh(mouse_logits) * self.mouse_scale

        action: ActionDict = {
            'forward': move_idx == 0,
            'back': move_idx == 1,
            'left': move_idx == 2,
            'right': move_idx == 3,
            'jump': bool(aux_samples[0].item()),
            'crouch': bool(aux_samples[1].item()),
            'walk': bool(aux_samples[2].item()),
            'fire': bool(aux_samples[3].item()),
            'mouse_dx': int(round(float(mouse_values[0].item()))),
            'mouse_dy': int(round(float(mouse_values[1].item()))),
        }

        self.logger.info(
            'NeuralRuntimeAgent outputs | move_probs=%s | aux_probs=%s | mouse=%s',
            [round(float(v), 4) for v in move_probs.detach().cpu().tolist()],
            [round(float(v), 4) for v in aux_probs.detach().cpu().tolist()],
            [round(float(v), 2) for v in mouse_values.detach().cpu().tolist()],
        )
        self.logger.info('NeuralRuntimeAgent action dict: %s', action)
        return action

class FullNeuralRuntimeAgent:
    def __init__(self, seed: int = 42, aim_checkpoint: str | None = None, movement_checkpoint: str | None = None, tracker_checkpoint: str | None = None) -> None:
        if not torch_available():
            raise RuntimeError('PyTorch is not available.')

        import torch
        from cs2_ai.ml.models.aim_attention import AimAttentionModel
        from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
        from cs2_ai.pipeline.neural_ai_pipeline import NeuralAIPipeline
        from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor
        from cs2_ai.features.movement_features import MovementFeatureExtractor
        from cs2_ai.features.aim_features import AimFeatureExtractor
        from cs2_ai.config import MAX_ENEMIES

        set_seed(seed)
        self.device = get_device()
        self.logger = logging.getLogger(__name__)

        self.aim_model = AimAttentionModel(input_dim=AimFeatureExtractor().feature_dim()).to(self.device)
        self.movement_model = DecisionDQN(input_dim=MovementFeatureExtractor().feature_dim(), action_dim=8).to(self.device)
        self.tracker_model = EnemyTrackerLSTM(input_dim=EnemyTrackerFeatureExtractor().feature_dim(), output_enemies=MAX_ENEMIES).to(self.device)

        if aim_checkpoint and Path(aim_checkpoint).exists():
            self.aim_model.load_state_dict(torch.load(aim_checkpoint, map_location=self.device).get('model_state_dict'))
        if movement_checkpoint and Path(movement_checkpoint).exists():
            self.movement_model.load_state_dict(torch.load(movement_checkpoint, map_location=self.device).get('model_state_dict'))
        if tracker_checkpoint and Path(tracker_checkpoint).exists():
            self.tracker_model.load_state_dict(torch.load(tracker_checkpoint, map_location=self.device).get('model_state_dict'))

        self.aim_model.eval()
        self.movement_model.eval()
        self.tracker_model.eval()

        self.pipeline = NeuralAIPipeline(self.aim_model, self.movement_model, self.tracker_model, device=self.device)
        self.logger.info('FullNeuralRuntimeAgent initialized.')

    def predict_state(self, game_state: GameState, _features: dict[str, float | int | bool] | None = None) -> ActionDict:
        from runtime_agent import PipelineRuntimeAgent
        
        # We need to convert GameState to AIGameState, then call pipeline.step()
        # The easiest way is to borrow PipelineRuntimeAgent's conversion methods
        agent_helper = PipelineRuntimeAgent()
        ai_state = agent_helper._to_ai_game_state(game_state)
        action_plan = self.pipeline.step(ai_state)
        action = agent_helper._action_plan_to_action_dict(action_plan)
        
        self.logger.info('FullNeuralRuntimeAgent action dict: %s', action)
        return action
