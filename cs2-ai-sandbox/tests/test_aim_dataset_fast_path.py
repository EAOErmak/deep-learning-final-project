import pytest
import pandas as pd
import numpy as np
from cs2_ai.ml.training.train_aim import AimSequenceTorchDataset

class DummySampleRef:
    def __init__(self, idx, perspective_steamid, ticks_to_check):
        self.idx = idx
        self.perspective_steamid = perspective_steamid
        self.tick_indices = ticks_to_check[:-1]
        self.target_tick = ticks_to_check[-1]

class MockSingleRoundDataset:
    def __init__(self, round_tick_rows, samples):
        self.round_tick_rows = round_tick_rows
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def get_sample_metadata(self, idx):
        s = self._samples[idx]
        return {
            'perspective_steamid': s.perspective_steamid,
            'tick_indices': s.tick_indices,
            'target_tick': s.target_tick
        }

class MockAimSequenceTorchDataset(AimSequenceTorchDataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.require_spotted_enemy = True
        
    def _valid_indices_cache_path(self):
        return None
        
    def sample_has_spotted_enemy(self, sample_metadata):
        # The slow path fallback mock for testing correctness
        perspective = int(sample_metadata['perspective_steamid'])
        ticks_to_check = list(sample_metadata.get('tick_indices', []))
        ticks_to_check.append(int(sample_metadata['target_tick']))
        
        for tick in ticks_to_check:
            for round_number, ticks in self.base_dataset.round_tick_rows.items():
                tick_df = ticks.get(tick)
                if tick_df is not None and not tick_df.empty:
                    if 'steamid' not in tick_df.columns:
                        continue
                    steamids = pd.to_numeric(tick_df['steamid'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
                    spotted_col = 'is_spotted' if 'is_spotted' in tick_df.columns else 'spotted'
                    if spotted_col not in tick_df.columns:
                        continue
                        
                    spotted_values = pd.to_numeric(tick_df[spotted_col], errors='coerce').fillna(0).to_numpy(dtype=np.float64) > 0
                    if 'is_alive' in tick_df.columns:
                        alive_values = tick_df['is_alive'].fillna(False).astype(bool).to_numpy(dtype=bool)
                    else:
                        alive_values = np.ones(len(tick_df), dtype=bool)
                        
                    self_mask = steamids == perspective
                    if not np.any(self_mask):
                        continue
                        
                    if 'team_num' in tick_df.columns:
                        team_values = pd.to_numeric(tick_df['team_num'], errors='coerce')
                        self_team = team_values.loc[self_mask].iloc[0]
                        enemy_mask = (team_values != self_team) & ~team_values.isna()
                    else:
                        enemy_mask = steamids != perspective
                        
                    if np.any(enemy_mask & spotted_values & alive_values):
                        return True
        return False


def test_fast_path_correctness():
    # Setup dummy tick rows
    # Tick 1: SteamID 100 (Team 1), SteamID 200 (Team 2, spotted=True), SteamID 300 (Team 2, spotted=False)
    # Tick 2: SteamID 100 (Team 1), SteamID 200 (Team 2, spotted=False, dead)
    
    df_tick1 = pd.DataFrame({
        'steamid': [100, 200, 300],
        'team_num': [1, 2, 2],
        'is_spotted': [0, 1, 0],
        'is_alive': [True, True, True]
    })
    
    df_tick2 = pd.DataFrame({
        'steamid': [100, 200],
        'team_num': [1, 2],
        'spotted': [0, 0],
        'is_alive': [True, False]
    })
    
    round_tick_rows = {
        1: {
            1: df_tick1,
            2: df_tick2
        }
    }
    
    samples = [
        DummySampleRef(0, 100, [1]), # Should be valid (100 sees 200)
        DummySampleRef(1, 100, [2]), # Should be invalid (200 is dead, spotted is 0)
        DummySampleRef(2, 200, [1]), # Should be invalid (200 doesn't see anyone spotted)
        DummySampleRef(3, 300, [1]), # Should be invalid
    ]
    
    ds = MockSingleRoundDataset(round_tick_rows, samples)
    aim_ds = MockAimSequenceTorchDataset(ds)
    
    # Run the real method
    valid_indices = aim_ds._build_valid_indices()
    
    # Verify
    assert valid_indices == [0]

def test_fast_path_fallback_missing_cols():
    # Missing 'steamid' column
    df_tick1 = pd.DataFrame({
        'team_num': [1, 2],
        'is_spotted': [1, 1]
    })
    
    round_tick_rows = { 1: { 1: df_tick1 } }
    samples = [DummySampleRef(0, 100, [1])]
    
    ds = MockSingleRoundDataset(round_tick_rows, samples)
    aim_ds = MockAimSequenceTorchDataset(ds)
    
    valid_indices = aim_ds._build_valid_indices()
    # It should fallback and return []
    assert valid_indices == []
