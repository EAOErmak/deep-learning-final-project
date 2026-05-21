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

    from cs2_ai.pipeline.neural_ai_pipeline import NeuralAIPipeline
else:
    torch = None
    NeuralAIPipeline = None


class MovementRuntimeDecodeTests(unittest.TestCase):
    def setUp(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch not available")
        self.pipeline = NeuralAIPipeline.__new__(NeuralAIPipeline)
        self.pipeline.movement_threshold = 0.5

    def test_decode_old_sequence_logits_uses_last_timestep(self):
        features = torch.zeros((1, 4, 3), dtype=torch.float32)
        output = torch.tensor([[[0.0] * 6, [0.0] * 6, [0.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype=torch.float32)
        logits = self.pipeline._normalize_movement_outputs(output, features)
        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertEqual(logits.squeeze(0).tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_decode_chunk_logits_uses_first_action(self):
        features = torch.zeros((1, 32, 3), dtype=torch.float32)
        output = torch.tensor([[[1.0] * 7, [2.0] * 7, [3.0] * 7]], dtype=torch.float32)
        logits = self.pipeline._normalize_movement_outputs(output, features)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertEqual(logits.squeeze(0).tolist(), [1.0] * 7)

    def test_decode_flat_logits_passthrough(self):
        features = torch.zeros((1, 4, 3), dtype=torch.float32)
        output = torch.tensor([[1.0, 0.0, -1.0, 2.0, -2.0, 3.0, 4.0]], dtype=torch.float32)
        logits = self.pipeline._normalize_movement_outputs(output, features)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertEqual(logits.squeeze(0).tolist(), [1.0, 0.0, -1.0, 2.0, -2.0, 3.0, 4.0])

    def test_jump_threshold_maps_to_should_jump(self):
        probs = torch.tensor([0.6, 0.4, 0.1, 0.2, 0.7, 0.8, 0.51], dtype=torch.float32)
        should_jump = bool(probs[6] > self.pipeline.movement_threshold)
        self.assertTrue(should_jump)


if __name__ == "__main__":
    unittest.main()
