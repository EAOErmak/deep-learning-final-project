import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from cs2_ai.dataset.route_grid_dataset import RouteGridSequenceDataset, collate_route_batch, denormalize_route_xyz

def inspect_dataset():
    manifest_path = "data/processed/rounds-dataset-grid-splits/train_rounds.txt"
    print(f"Loading dataset from {manifest_path}...")
    
    dataset = RouteGridSequenceDataset(
        manifest_path=manifest_path,
        history_len=6,
        min_subseq_len=4,
        max_subseq_len=8,
        samples_per_epoch=10,
        route_output_mode="xyz"
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_route_batch)
    
    batch = next(iter(loader))
    
    print("\n" + "="*60)
    print("INSPECTING ONE SAMPLE FROM THE ROUTE DATASET (XYZ MODE)")
    print("="*60)
    
    idx = 0
    
    meta = batch["metas"][idx]
    print(f"\n[META INFORMATION]")
    for k, v in meta.items():
        print(f"  {k}: {v}")
        
    print("\n[NORMALIZED TENSORS (Input to the Neural Network)]")
    print(f"  Current XYZ: {[round(x, 4) for x in batch['current_xyz'][idx].tolist()]}")
    print(f"  Target XYZ:  {[round(x, 4) for x in batch['target_xyz'][idx].tolist()]}")
    print(f"  Next XYZ:    {[round(x, 4) for x in batch['next_xyz'][idx].tolist()]}")
    
    print("\n  History Sequence (Normalized):")
    hist_mask = batch["history_mask"][idx]
    hist_xyz = batch["history_xyz"][idx]
    
    for i in range(len(hist_mask)):
        is_real = hist_mask[i].item() == 1.0
        status = "REAL" if is_real else "PAD "
        vals = [round(x, 4) for x in hist_xyz[i].tolist()]
        print(f"    Step {i} [{status}]: {vals}")
        
    print("\n" + "-"*60)
    print("[WORLD UNITS (Real Game Coordinates)]")
    print("-"*60)
    
    curr_wu = denormalize_route_xyz(batch['current_xyz'][idx])
    targ_wu = denormalize_route_xyz(batch['target_xyz'][idx])
    next_wu = denormalize_route_xyz(batch['next_xyz'][idx])
    
    print(f"  Current XYZ: [{curr_wu[0]:.1f}, {curr_wu[1]:.1f}, {curr_wu[2]:.1f}]")
    print(f"  Target XYZ:  [{targ_wu[0]:.1f}, {targ_wu[1]:.1f}, {targ_wu[2]:.1f}]")
    print(f"  Next XYZ:    [{next_wu[0]:.1f}, {next_wu[1]:.1f}, {next_wu[2]:.1f}]")
    
    print("\n  History Sequence (World Units):")
    for i in range(len(hist_mask)):
        is_real = hist_mask[i].item() == 1.0
        status = "REAL" if is_real else "PAD "
        h_wu = denormalize_route_xyz(hist_xyz[i])
        print(f"    Step {i} [{status}]: [{h_wu[0]:.1f}, {h_wu[1]:.1f}, {h_wu[2]:.1f}]")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    inspect_dataset()
