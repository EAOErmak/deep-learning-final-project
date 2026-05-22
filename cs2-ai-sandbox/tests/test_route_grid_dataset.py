import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.create_grid_round_split import create_grid_round_split
from cs2_ai.dataset.route_grid_dataset import RouteGridSequenceDataset, collate_route_batch

@pytest.fixture
def fake_dataset_root(tmp_path):
    dataset_root = tmp_path / "rounds-dataset-grid"
    
    # Create 2 fake demos, 2 rounds each
    demos = ["demo1", "demo2"]
    for i, demo in enumerate(demos):
        round_dir = dataset_root / demo / "rounds"
        round_dir.mkdir(parents=True)
        
        for j in range(2):
            df = pd.DataFrame({
                "tick": [100, 101, 102, 103, 104, 105, 106, 107],
                "player_steamid": [1, 1, 1, 1, 2, 2, 2, 2],
                "block_id": [10, 10, 11, 12, 5, 5, 5, 6]
            })
            df.to_parquet(round_dir / f"round_{j}.parquet")
            
    return dataset_root

def test_create_grid_round_split(fake_dataset_root, tmp_path, monkeypatch):
    output_dir = tmp_path / "splits"
    
    # Mock sys.argv
    import sys
    test_args = [
        "script",
        "--dataset-root", str(fake_dataset_root),
        "--output-dir", str(output_dir),
        "--val-ratio", "0.5",
        "--split-mode", "round",
        "--seed", "42"
    ]
    monkeypatch.setattr(sys, 'argv', test_args)
    
    create_grid_round_split()
    
    train_manifest = output_dir / "train_rounds.txt"
    val_manifest = output_dir / "val_rounds.txt"
    
    assert train_manifest.exists()
    assert val_manifest.exists()
    
    with open(train_manifest) as f:
        train_lines = f.readlines()
    with open(val_manifest) as f:
        val_lines = f.readlines()
        
    assert len(train_lines) + len(val_lines) == 4

def test_route_grid_dataset(fake_dataset_root, tmp_path, monkeypatch):
    output_dir = tmp_path / "splits"
    
    import sys
    test_args = [
        "script",
        "--dataset-root", str(fake_dataset_root),
        "--output-dir", str(output_dir),
        "--val-ratio", "0.5",
        "--split-mode", "round"
    ]
    monkeypatch.setattr(sys, 'argv', test_args)
    create_grid_round_split()
    
    train_manifest = output_dir / "train_rounds.txt"
    
    dataset = RouteGridSequenceDataset(
        manifest_path=train_manifest,
        history_len=4,
        min_subseq_len=2,
        max_subseq_len=4,
        samples_per_epoch=10,
        fixed_samples=False
    )
    
    assert len(dataset) == 10
    
    # test consecutive duplicates removed
    # player 1: [10, 10, 11, 12] -> [10, 11, 12]
    # player 2: [5, 5, 5, 6] -> [5, 6]
    # but since length is 3 and 2, and history+min_goal = 4+1 = 5, they will be filtered!
    
    # Let's write a bigger single parquet file
    big_parquet_dir = tmp_path / "big_dataset" / "demo1" / "rounds"
    big_parquet_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "tick": list(range(100, 110)),
        "player_steamid": [1]*10,
        "block_id": [10, 10, 11, 12, 13, 14, 15, 16, 17, 18] # len = 9
    })
    df.to_parquet(big_parquet_dir / "round_0.parquet")
    
    big_manifest = tmp_path / "big_manifest.txt"
    with open(big_manifest, "w") as f:
        f.write(str(big_parquet_dir / "round_0.parquet") + "\n")
        
    ds = RouteGridSequenceDataset(
        manifest_path=big_manifest,
        history_len=4,
        min_subseq_len=4,
        max_subseq_len=6,
        samples_per_epoch=5
    )
    
    # One trajectory valid
    assert len(ds.trajectories) == 1
    
    # Random subsequence bounds test
    for _ in range(50):
        item = ds[0]
        meta = item["meta"]
        L = meta["subseq_len"]
        assert 4 <= L <= 6
        
        target_raw = meta["route_target_raw"]
        assert item["target_block"].item() == ds.encode_block(target_raw)
        
        assert item["history_blocks"].shape == (4,)
        assert isinstance(item["current_block"], torch.Tensor)
        assert isinstance(item["target_block"], torch.Tensor)
        assert isinstance(item["next_block"], torch.Tensor)
        
        # Check current == history[-1]
        assert item["history_blocks"][-1].item() == item["current_block"].item()
        
    # Fixed validation stability test
    ds_fixed = RouteGridSequenceDataset(
        manifest_path=big_manifest,
        history_len=4,
        min_subseq_len=4,
        max_subseq_len=6,
        fixed_samples=True
    )
    item1 = ds_fixed[0]
    item2 = ds_fixed[0]
    assert item1["meta"] == item2["meta"]
    assert torch.equal(item1["history_blocks"], item2["history_blocks"])
    assert item1["current_block"].item() == item2["current_block"].item()
    assert item1["target_block"].item() == item2["target_block"].item()
    assert item1["next_block"].item() == item2["next_block"].item()
    
def test_collate_function():
    batch = [
        {
            "history_blocks": torch.tensor([0, 1, 2, 3]),
            "current_block": torch.tensor(3),
            "target_block": torch.tensor(5),
            "next_block": torch.tensor(4),
            "meta": {"id": 1}
        },
        {
            "history_blocks": torch.tensor([0, 0, 1, 2]),
            "current_block": torch.tensor(2),
            "target_block": torch.tensor(6),
            "next_block": torch.tensor(3),
            "meta": {"id": 2}
        }
    ]
    
    hist, curr, targ, next_b, metas = collate_route_batch(batch)
    assert hist.shape == (2, 4)
    assert curr.shape == (2,)
    assert targ.shape == (2,)
    assert next_b.shape == (2,)
    assert len(metas) == 2
