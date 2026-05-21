from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.features.enemy_tracker_features import TRACKER_FEATURE_NAMES, EnemyTrackerFeatureExtractor
from cs2_ai.ml.models.enemy_tracker_lstm import (
    ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK,
    ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK,
)
from cs2_ai.ml.training.train_enemy_tracker import EnemyTrackerSequenceTorchDataset
from cs2_ai.ml.utils.torch_utils import torch_available
from cs2_ai.schemas.game_state import GameStateSequence
from cs2_ai.state.game_state_builder import GameStateBuilder
from tests.test_data_contract_parity import make_tick_rows

if torch_available():
    import torch
    from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
else:
    torch = None
    EnemyTrackerLSTM = None


class InMemoryEnemyTrackerBaseDataset:
    def __init__(self):
        self.seq_len = 2
        self.builder = GameStateBuilder()
        visible_state = self.builder.build_from_tick_rows(make_tick_rows(visible_enemy=True, tick=100), perspective_steamid=1)
        hidden_state = self.builder.build_from_tick_rows(make_tick_rows(visible_enemy=False, tick=101), perspective_steamid=1)
        self.sequence = GameStateSequence(perspective_steamid=1, states=[visible_state, hidden_state])
        self.metadata = {
            'sample_id': 'demo::r1::p1::s100::t102',
            'demo_name': 'demo',
            'parquet_path': 'demo.parquet',
            'round_number': 1,
            'perspective_steamid': 1,
            'tick_indices': (100, 101),
            'target_tick': 102,
        }
        self.truth_rows = {
            101: make_tick_rows(visible_enemy=False, tick=101),
            102: make_tick_rows(visible_enemy=False, tick=102),
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return SimpleNamespace(sequence=self.sequence)

    def get_sample_metadata(self, idx):
        return dict(self.metadata)

    def build_truth_state_for_sample_tick(self, sample_metadata, tick: int):
        return self.builder.build_truth_from_tick_rows(self.truth_rows[int(tick)], perspective_steamid=1)


class EnemyTrackerLSTMModesTests(unittest.TestCase):
    def test_enemy_tracker_lstm_each_tick_shape(self):
        if not torch_available():
            self.skipTest('PyTorch not available')
        model = EnemyTrackerLSTM(
            input_dim=64,
            hidden_dim=32,
            num_layers=2,
            output_enemies=5,
            dropout=0.1,
            output_mode=ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK,
        )
        x = torch.zeros((3, 7, 64), dtype=torch.float32)
        positions, confidences = model(x)
        self.assertEqual(tuple(positions.shape), (3, 7, 5, 3))
        self.assertEqual(tuple(confidences.shape), (3, 7, 5))

    def test_enemy_tracker_lstm_target_tick_shape(self):
        if not torch_available():
            self.skipTest('PyTorch not available')
        model = EnemyTrackerLSTM(
            input_dim=64,
            hidden_dim=32,
            num_layers=2,
            output_enemies=5,
            dropout=0.1,
            output_mode=ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK,
        )
        x = torch.zeros((3, 7, 64), dtype=torch.float32)
        positions, confidences = model(x)
        self.assertEqual(tuple(positions.shape), (3, 5, 3))
        self.assertEqual(tuple(confidences.shape), (3, 5))

    def test_tracker_dataset_shapes_for_both_modes(self):
        dataset_each = EnemyTrackerSequenceTorchDataset(
            InMemoryEnemyTrackerBaseDataset(),
            seq_len=2,
            output_mode=ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK,
        )
        features, target_positions, target_confidences, age_bucket_ids, _ = dataset_each[0]
        self.assertEqual(tuple(features.shape), (2, EnemyTrackerFeatureExtractor(seq_len=2).feature_dim()))
        self.assertEqual(tuple(target_positions.shape), (2, 5, 3))
        self.assertEqual(tuple(target_confidences.shape), (2, 5))
        self.assertEqual(tuple(age_bucket_ids.shape), (2, 5))

        dataset_target = EnemyTrackerSequenceTorchDataset(
            InMemoryEnemyTrackerBaseDataset(),
            seq_len=2,
            output_mode=ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK,
        )
        _, target_positions_last, target_confidences_last, age_bucket_ids_last, _ = dataset_target[0]
        self.assertEqual(tuple(target_positions_last.shape), (5, 3))
        self.assertEqual(tuple(target_confidences_last.shape), (5,))
        self.assertEqual(tuple(age_bucket_ids_last.shape), (5,))

    def test_tracker_features_do_not_leak_hidden_truth_positions(self):
        builder = GameStateBuilder()
        hidden_state = builder.build_from_tick_rows(make_tick_rows(visible_enemy=False, tick=100), perspective_steamid=1)
        sequence = GameStateSequence(perspective_steamid=1, states=[hidden_state])
        features = EnemyTrackerFeatureExtractor(seq_len=1).extract(sequence)
        name_to_idx = {name: idx for idx, name in enumerate(TRACKER_FEATURE_NAMES)}
        frame = features[-1]
        self.assertEqual(float(frame[name_to_idx['enemy_0_rel_x']]), 0.0)
        self.assertEqual(float(frame[name_to_idx['enemy_0_rel_y']]), 0.0)
        self.assertEqual(float(frame[name_to_idx['enemy_0_rel_z']]), 0.0)
        self.assertEqual(float(frame[name_to_idx['enemy_0_visible_mask']]), 0.0)
        self.assertEqual(float(frame[name_to_idx['enemy_0_last_seen_mask']]), 0.0)
        self.assertEqual(float(frame[name_to_idx['enemy_0_unavailable_mask']]), 1.0)


if __name__ == '__main__':
    unittest.main()
