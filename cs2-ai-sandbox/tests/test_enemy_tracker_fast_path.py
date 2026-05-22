import pytest
import numpy as np
import pandas as pd
from cs2_ai.ml.training.train_enemy_tracker import EnemyTrackerSequenceTorchDataset, MAX_ENEMIES

class MockFeatureExtractor:
    def feature_dim(self):
        return 10
    def extract(self, seq):
        return np.zeros((1, 10))

class MockBaseDataset:
    def __init__(self, round_tick_rows):
        self.round_tick_rows = round_tick_rows
        self.samples = [{"tick_indices": [10], "target_tick": 10, "perspective_steamid": 100, "round_number": 1, "demo_name": "demo1"}]

    def get_sample_metadata(self, idx):
        return self.samples[idx]

class MockSingleRoundDataset(MockBaseDataset):
    pass

def test_fast_target_path_parity():
    # Setup dataframe
    df = pd.DataFrame({
        'steamid': [100, 200, 300, 400],
        'team_num': [1, 2, 2, 1], # 200 and 300 are enemies to 100
        'is_alive': [True, True, False, True], # 300 is dead
        'X': [0.0, 10.0, 20.0, 30.0],
        'Y': [0.0, 11.0, 21.0, 31.0],
        'Z': [0.0, 12.0, 22.0, 32.0],
        'is_spotted': [1, 1, 1, 1] # shouldn't matter
    })
    
    ds = MockSingleRoundDataset({1: {10: df}})
    
    # We will just test _build_fast_target directly since we mocked the dataset
    tracker_ds = EnemyTrackerSequenceTorchDataset(ds, seq_len=1)
    tracker_ds.feature_extractor = MockFeatureExtractor()
    
    roster_steamids = [200, 300, 500] + [0] * (MAX_ENEMIES - 3)
    
    pos, conf = tracker_ds._build_fast_target(ds, round_number=1, tick=10, perspective_steamid=100, roster_steamids=roster_steamids)
    
    assert pos is not None
    assert conf is not None
    
    # Check enemy 200 (index 0 in roster) -> alive, enemy
    assert conf[0] == 1.0
    assert np.allclose(pos[0], [10.0, 11.0, 12.0])
    
    # Check enemy 300 (index 1 in roster) -> dead, enemy
    assert conf[1] == 0.0 # because is_alive is False
    assert np.allclose(pos[1], [20.0, 21.0, 22.0])
    
    # Check enemy 500 (index 2 in roster) -> not in tick
    assert conf[2] == 0.0
    
    # Check teammate 400 -> not in roster, but also not enemy, shouldn't affect conf
    
def test_fallback_missing_cols():
    df = pd.DataFrame({
        'steamid': [100, 200],
        'team_num': [1, 2]
        # missing X, Y, Z
    })
    ds = MockSingleRoundDataset({1: {10: df}})
    tracker_ds = EnemyTrackerSequenceTorchDataset(ds, seq_len=1)
    roster_steamids = [200] + [0] * (MAX_ENEMIES - 1)
    pos, conf = tracker_ds._build_fast_target(ds, round_number=1, tick=10, perspective_steamid=100, roster_steamids=roster_steamids)
    
    assert pos is None
    assert conf is None

