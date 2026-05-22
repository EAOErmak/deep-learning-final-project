import pytest
import numpy as np
from cs2_ai.ml.training.train_movement import MovementSequenceTorchDataset, compute_action_ratios, collect_expected_demo_counts

class DummySampleRef:
    def __init__(self, demo_name, round_number, target_tick):
        self.demo_name = demo_name
        self.round_number = round_number
        self.target_tick = target_tick
        self.tick_indices = [target_tick - 1]

class MockMovementBaseDataset:
    def __init__(self, round_tick_rows, samples):
        self.round_tick_rows = round_tick_rows
        self.samples = samples
        self.seq_len = 1
        self.action_dim = 2
        self.action_names = ["a", "b"]

    def __len__(self):
        return len(self.samples)

    def get_sample_metadata(self, idx):
        s = self.samples[idx]
        return {
            'sample_id': f"{s.demo_name}_{idx}",
            'demo_name': s.demo_name,
            'round_number': s.round_number,
            'target_tick': s.target_tick,
            'tick_indices': s.tick_indices,
            'perspective_steamid': 123
        }
        
    def __getitem__(self, idx):
        class MockSeqSample:
            sequence = None
        return MockSeqSample()

class MockMovementSequenceTorchDataset(MovementSequenceTorchDataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.target_mode = "action_chunk"
        self.chunk_len = 2
        self.action_names = base_dataset.action_names
        self.movement_feature_mode = "legacy"
        self.use_grid_navigation_features = False
        self.valid_indices = self._build_valid_indices()
        
    def _build_valid_indices(self):
        return list(range(len(self.base_dataset)))

    def _resolve_round_tick_rows(self, metadata):
        return self.base_dataset.round_tick_rows[metadata['round_number']]
        
    def _build_target_for_ticks(self, metadata, ticks):
        return np.ones((len(ticks), 2))

def test_resolve_target_ticks_cache_collision():
    # Two demos, same round number, different ticks
    round_tick_rows_demo1 = {
        1: { 10: None, 11: None, 12: None }
    }
    round_tick_rows_demo2 = {
        1: { 20: None, 21: None, 22: None }
    }
    
    samples1 = [DummySampleRef("demo1", 1, 10)]
    samples2 = [DummySampleRef("demo2", 1, 20)]
    
    # We will mock the base dataset to return different rows based on demo_name
    class MultiDemoMock(MockMovementBaseDataset):
        def _resolve_round_tick_rows(self, metadata):
            if metadata['demo_name'] == "demo1":
                return round_tick_rows_demo1[metadata['round_number']]
            return round_tick_rows_demo2[metadata['round_number']]
            
    base_ds = MultiDemoMock({}, samples1 + samples2)
    # Monkeypatch to avoid the standard resolve
    ds = MockMovementSequenceTorchDataset(base_ds)
    ds._resolve_round_tick_rows = base_ds._resolve_round_tick_rows
    
    ticks1 = ds._resolve_target_ticks(base_ds.get_sample_metadata(0))
    ticks2 = ds._resolve_target_ticks(base_ds.get_sample_metadata(1))
    
    assert ticks1 == [10, 11]
    assert ticks2 == [20, 21]

def test_compute_action_ratios_sampling():
    samples = [DummySampleRef("demo", 1, 10 + i) for i in range(100)]
    round_tick_rows = {1: {10 + i: None for i in range(110)}}
    base_ds = MockMovementBaseDataset(round_tick_rows, samples)
    ds = MockMovementSequenceTorchDataset(base_ds)
    
    # sample size 10
    stats1 = compute_action_ratios(ds, sample_size=10, seed=42)
    assert stats1['total_steps'] == 10 * 2  # 10 samples * chunk_len 2
    
    # sample size 0 (full scan)
    stats2 = compute_action_ratios(ds, sample_size=0, seed=42)
    assert stats2['total_steps'] == 100 * 2

def test_collect_expected_demo_counts_subset():
    samples = [
        DummySampleRef("demo1", 1, 10),
        DummySampleRef("demo1", 1, 11),
        DummySampleRef("demo2", 1, 20),
    ]
    base_ds = MockMovementBaseDataset({}, samples)
    
    class MockSubset:
        def __init__(self, ds, indices):
            self.dataset = ds
            self.indices = indices
            
    subset = MockSubset(base_ds, [0, 2])
    counts = collect_expected_demo_counts(subset)
    assert counts == {"demo1": 1, "demo2": 1}


def test_compute_pos_weight_uses_provided_stats(monkeypatch):
    import cs2_ai.ml.training.train_movement as tm
    called = False
    
    def mock_compute_action_ratios(*args, **kwargs):
        nonlocal called
        called = True
        return {'positive_ratios': [0.5, 0.5]}
        
    monkeypatch.setattr(tm, "compute_action_ratios", mock_compute_action_ratios)
    
    samples = [DummySampleRef("demo", 1, 10)]
    base_ds = MockMovementBaseDataset({}, samples)
    ds = MockMovementSequenceTorchDataset(base_ds)
    
    provided_stats = {
        'positive_ratios': [0.1, 0.2]
    }
    
    pos_weight = tm.compute_pos_weight(
        ds, mode="auto", explicit_values=None, stats=provided_stats
    )
    
    assert not called
    np.testing.assert_allclose(pos_weight, [9.0, 4.0])
