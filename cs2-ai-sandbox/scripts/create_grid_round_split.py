import argparse
import logging
import random
from pathlib import Path
from collections import defaultdict

def create_grid_round_split():
    parser = argparse.ArgumentParser(description="Create train/val split for grid-block round dataset.")
    parser.add_argument("--dataset-root", type=str, default="data/processed/rounds-dataset-grid",
                        help="Root directory of the dataset.")
    parser.add_argument("--output-dir", type=str, default="data/processed/rounds-dataset-grid-splits",
                        help="Directory to save the manifest files.")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Ratio of validation samples.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting.")
    parser.add_argument("--split-mode", type=str, choices=["round", "demo"], default="round",
                        help="Splitting mode: by 'round' or by 'demo' directory.")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    
    if not dataset_root.exists():
        logging.error(f"Dataset root does not exist: {dataset_root}")
        return
        
    # Find all parquet files
    # The pattern is: <dataset-root>/*/rounds/round_*.parquet
    parquet_files = list(dataset_root.glob("*/rounds/round_*.parquet"))
    
    if not parquet_files:
        logging.error(f"No parquet files found in {dataset_root} matching '*/rounds/round_*.parquet'.")
        return
        
    # Sort for determinism before shuffling
    parquet_files.sort()
    
    rng = random.Random(args.seed)
    
    train_files = []
    val_files = []
    
    if args.split_mode == "round":
        rng.shuffle(parquet_files)
        total = len(parquet_files)
        val_count = max(1, int(total * args.val_ratio))
        val_files = parquet_files[:val_count]
        train_files = parquet_files[val_count:]
        
    elif args.split_mode == "demo":
        demo_to_rounds = defaultdict(list)
        for p in parquet_files:
            # p.parent is 'rounds', p.parent.parent is the demo dir
            demo_name = p.parent.parent.name
            demo_to_rounds[demo_name].append(p)
            
        demos = list(demo_to_rounds.keys())
        demos.sort()
        rng.shuffle(demos)
        
        total_demos = len(demos)
        val_demos_count = max(1, int(total_demos * args.val_ratio))
        
        val_demos = set(demos[:val_demos_count])
        
        for p in parquet_files:
            demo_name = p.parent.parent.name
            if demo_name in val_demos:
                val_files.append(p)
            else:
                train_files.append(p)
                
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = output_dir / "train_rounds.txt"
    val_manifest = output_dir / "val_rounds.txt"
    
    with open(train_manifest, "w", encoding="utf-8") as f:
        for p in sorted(train_files):
            f.write(f"{p.absolute().as_posix()}\n")
            
    with open(val_manifest, "w", encoding="utf-8") as f:
        for p in sorted(val_files):
            f.write(f"{p.absolute().as_posix()}\n")
            
    logging.info(f"Total rounds: {len(parquet_files)}")
    logging.info(f"Train rounds: {len(train_files)}")
    logging.info(f"Val rounds: {len(val_files)}")
    logging.info(f"Split mode: {args.split_mode}")
    logging.info(f"Output saved to:\n  {train_manifest.as_posix()}\n  {val_manifest.as_posix()}")

if __name__ == "__main__":
    create_grid_round_split()
