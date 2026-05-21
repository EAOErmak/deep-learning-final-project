from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch
    from cs2_ai.ml.models.movement_gru import MovementGRU
    from cs2_ai.ml.training.train_movement import build_model, MOVEMENT_MODEL_DECISION_DQN, MOVEMENT_MODEL_GRU
else:
    torch = None
    MovementGRU = None


class MovementGRUTests(unittest.TestCase):
    def test_movement_gru_output_shape(self):
        if not torch_available():
            self.skipTest("PyTorch not available")
        model = MovementGRU(input_dim=37, action_dim=7, chunk_len=8, hidden_dim=32, num_layers=2, dropout=0.1)
        x = torch.zeros((4, 16, 37), dtype=torch.float32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, 8, 7))

    def test_build_model_keeps_decision_dqn_available(self):
        if not torch_available():
            self.skipTest("PyTorch not available")
        dqn = build_model(
            model_name=MOVEMENT_MODEL_DECISION_DQN,
            input_dim=37,
            action_dim=7,
            hidden_dim=32,
            target_len=8,
            gru_num_layers=2,
            gru_dropout=0.1,
            device='cpu',
        )
        gru = build_model(
            model_name=MOVEMENT_MODEL_GRU,
            input_dim=37,
            action_dim=7,
            hidden_dim=32,
            target_len=8,
            gru_num_layers=2,
            gru_dropout=0.1,
            device='cpu',
        )
        self.assertNotEqual(type(dqn).__name__, type(gru).__name__)
        self.assertEqual(type(gru).__name__, 'MovementGRU')


if __name__ == "__main__":
    unittest.main()
