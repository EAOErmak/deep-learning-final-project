from __future__ import annotations

import logging
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
        'self_hp',
        'self_money',
        'ammo',
        'yaw',
        'pitch',
        'enemy_visible',
        'enemy_rel_x',
        'enemy_rel_y',
        'enemy_rel_z',
        'enemy_hp',
        'enemy_distance',
    ]

    def __init__(self, seed: int = 42, mouse_scale: float = 18.0) -> None:
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
        self.logger.info('NeuralRuntimeAgent initialized | device=%s | seed=%s', self.device, seed)

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
