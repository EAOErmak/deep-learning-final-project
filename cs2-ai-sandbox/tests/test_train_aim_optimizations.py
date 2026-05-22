import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

from cs2_ai.ml.training.train_aim import (
    compute_binary_action_stats,
    compute_binary_pos_weight,
    collect_expected_demo_counts,
)
from cs2_ai.ml.training.training_dataset_utils import filter_dataset_by_trained_rounds

class MockTarget:
    def __init__(self, binary_actions, valid_aim_mask):
        self.binary_actions = np.array(binary_actions, dtype=np.float64)
        self.valid_aim_mask = np.array(valid_aim_mask, dtype=np.float64)

class MockDataset:
    def __init__(self, count, samples=None):
        self.count = count
        self.samples = samples
        self.build_target_calls = 0
        self.metadata_calls = 0

    def __len__(self):
        return self.count

    def get_sample_metadata(self, idx):
        self.metadata_calls += 1
        return {'demo_name': f'demo_{idx % 2}', 'round_uid': f'uid_{idx}'}

    def build_target(self, sample_metadata):
        self.build_target_calls += 1
        return MockTarget([1, 0, 0], [1])


class MockSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)


def test_compute_binary_action_stats_sampling():
    ds = MockDataset(100)
    
    # Test 1: sample_size > dataset size
    stats1 = compute_binary_action_stats(ds, sample_size=150)
    assert stats1['sample_count_used'] == 100
    assert stats1['sample_count_total'] == 100
    assert not stats1['sampled']
    assert ds.build_target_calls == 100

    ds.build_target_calls = 0

    # Test 2: sample_size <= 0
    stats2 = compute_binary_action_stats(ds, sample_size=0)
    assert stats2['sample_count_used'] == 100
    assert stats2['sample_count_total'] == 100
    assert not stats2['sampled']
    assert ds.build_target_calls == 100

    ds.build_target_calls = 0

    # Test 3: Valid sampling
    stats3 = compute_binary_action_stats(ds, sample_size=20)
    assert stats3['sample_count_used'] == 20
    assert stats3['sample_count_total'] == 100
    assert stats3['sampled']
    assert ds.build_target_calls == 20


def test_compute_binary_pos_weight_reuse_stats():
    ds = MockDataset(10)
    
    # Passing predefined stats should avoid calling build_target
    predefined_stats = {
        'positive_ratios': [0.1, 0.2, 0.5],
        'positive_counts': [1, 2, 5],
        'valid_aim_rate': 1.0,
        'sample_count_used': 10,
        'sample_count_total': 10,
        'sampled': False,
        'binary_stats_sample_size': 50000,
    }
    
    pos_weight = compute_binary_pos_weight(ds, mode='auto', stats=predefined_stats)
    
    assert pos_weight is not None
    assert ds.build_target_calls == 0
    assert len(pos_weight) == 3


def test_collect_expected_demo_counts():
    samples = [{'demo_name': f'demo_{i % 2}'} for i in range(10)]
    ds = MockDataset(10, samples=samples)
    
    # Test 1: Using dataset.samples fast path
    counts = collect_expected_demo_counts(ds)
    assert counts == {'demo_0': 5, 'demo_1': 5}
    assert ds.metadata_calls == 0  # Should use fast path

    # Test 2: With Subset
    subset = MockSubset(ds, [0, 1, 2, 3])
    counts_subset = collect_expected_demo_counts(subset)
    assert counts_subset == {'demo_0': 2, 'demo_1': 2}
    assert ds.metadata_calls == 0


def test_filter_dataset_by_trained_rounds(tmp_path):
    ledger_path = tmp_path / 'ledger.json'
    
    class MockLedger:
        @staticmethod
        def load(path):
            return MockLedger()
            
        def read_trained_round_uids(self, module_name, model_name, checkpoint_path, match_mode):
            return {'uid_1', 'uid_3'}

    import cs2_ai.ml.training.training_dataset_utils as tdu
    original_ledger = tdu.TrainingRoundLedger
    tdu.TrainingRoundLedger = MockLedger

    ds = MockDataset(5)
    
    try:
        filtered, meta = filter_dataset_by_trained_rounds(
            ds,
            ledger_path=ledger_path,
            module_name='test',
            model_name='test',
            checkpoint_path='test',
            match_mode='module'
        )
        
        # uid_1 and uid_3 are trained, so we keep idx 0, 2, 4 (uid_0, uid_2, uid_4)
        assert filtered.indices == [0, 2, 4]
        assert meta['total_rounds_before_skip'] == 5
        assert meta['skipped_rounds_count'] == 2
        assert meta['remaining_rounds_count'] == 3
    finally:
        tdu.TrainingRoundLedger = original_ledger
