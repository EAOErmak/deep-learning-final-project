import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cs2_ai.dataset.route_grid_dataset import RouteGridSequenceDataset, collate_route_batch, SingleTrajectoryDataset

logger = logging.getLogger(__name__)

class RouteGRUModel(nn.Module):
    def __init__(self, num_blocks: int, embedding_dim: int = 128, hidden_dim: int = 256, pad_id: int = 0):
        super().__init__()
        self.num_blocks = num_blocks
        self.pad_id = pad_id
        
        self.block_emb = nn.Embedding(num_blocks, embedding_dim, padding_idx=pad_id)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_blocks)
        )
        
    def forward(self, history_blocks, current_block, target_block):
        # history_blocks: [B, H]
        # current_block: [B]
        # target_block: [B]
        
        hist_emb = self.block_emb(history_blocks) # [B, H, E]
        curr_emb = self.block_emb(current_block)  # [B, E]
        targ_emb = self.block_emb(target_block)   # [B, E]
        
        gru_out, hidden = self.gru(hist_emb)
        # hidden: [1, B, hidden_dim]
        hist_rep = hidden.squeeze(0) # [B, hidden_dim]
        
        # Concat
        combined = torch.cat([hist_rep, curr_emb, targ_emb], dim=-1)
        
        logits = self.mlp(combined) # [B, num_blocks]
        return logits

def train_route():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=str, default="data/processed/rounds-dataset-grid-splits/train_rounds.txt")
    parser.add_argument("--val-manifest", type=str, default="data/processed/rounds-dataset-grid-splits/val_rounds.txt")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--min-subseq-len", type=int, default=4)
    parser.add_argument("--max-subseq-len", type=int, default=32)
    parser.add_argument("--transitions-per-subsequence", type=int, default=1)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--max-eval-samples", type=int, default=10000)
    parser.add_argument("--max-train-rounds", type=int, default=None)
    parser.add_argument("--max-val-rounds", type=int, default=None)
    parser.add_argument("--shuffle-round-files", action="store_true")
    parser.add_argument("--round-file-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/route_gru.pt")
    
    # Debug / Overfit Mode Args
    parser.add_argument("--single-round-file", type=str, default=None)
    parser.add_argument("--player-steamid", type=str, default=None)
    parser.add_argument("--single-player-mode", action="store_true")
    parser.add_argument("--alive-only", action="store_true")
    parser.add_argument("--collapse-duplicates", action="store_true")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.single_player_mode and args.single_round_file:
        logger.info(f"--- DEBUG/OVERFIT MODE ---")
        logger.info(f"Loading single trajectory dataset from: {args.single_round_file}")
        
        train_dataset = SingleTrajectoryDataset(
            parquet_file=args.single_round_file,
            steamid=args.player_steamid,
            alive_only=args.alive_only,
            collapse_duplicates=args.collapse_duplicates,
            history_len=args.history_len,
            min_subseq_len=args.min_subseq_len,
            max_subseq_len=args.max_subseq_len,
            transitions_per_subsequence=args.transitions_per_subsequence,
            split="train",
            seed=args.seed
        )
        
        val_dataset = SingleTrajectoryDataset(
            parquet_file=args.single_round_file,
            steamid=args.player_steamid,
            alive_only=args.alive_only,
            collapse_duplicates=args.collapse_duplicates,
            history_len=args.history_len,
            min_subseq_len=args.min_subseq_len,
            max_subseq_len=args.max_subseq_len,
            transitions_per_subsequence=args.transitions_per_subsequence,
            split="val",
            seed=args.seed
        )
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info("Example Sample:")
            logger.info(f"  history_blocks: {sample['history_blocks'].tolist()}")
            logger.info(f"  current_block: {sample['current_block'].item()}")
            logger.info(f"  target_block: {sample['target_block'].item()}")
            logger.info(f"  next_block: {sample['next_block'].item()}")
            
    else:
        logger.info(f"history_len: {args.history_len}")
        logger.info(f"min_subseq_len: {args.min_subseq_len}")
        logger.info(f"max_subseq_len: {args.max_subseq_len}")
        logger.info(f"transitions_per_subsequence: {args.transitions_per_subsequence}")
        logger.info(f"samples_per_epoch: {args.samples_per_epoch}")
        logger.info(f"max_eval_samples: {args.max_eval_samples}")
        
        logger.info("Loading train dataset...")
        train_dataset = RouteGridSequenceDataset(
            manifest_path=args.train_manifest,
            history_len=args.history_len,
            min_subseq_len=args.min_subseq_len,
            max_subseq_len=args.max_subseq_len,
            transitions_per_subsequence=args.transitions_per_subsequence,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
            fixed_samples=False,
            max_rounds=args.max_train_rounds,
            shuffle_rounds=args.shuffle_round_files
        )
        
        logger.info("Loading val dataset...")
        val_dataset = RouteGridSequenceDataset(
            manifest_path=args.val_manifest,
            history_len=args.history_len,
            min_subseq_len=args.min_subseq_len,
            max_subseq_len=args.max_subseq_len,
            transitions_per_subsequence=args.transitions_per_subsequence,
            seed=args.seed + 1,
            fixed_samples=True,
            max_eval_samples=args.max_eval_samples,
            max_rounds=args.max_val_rounds,
            shuffle_rounds=args.shuffle_round_files
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_route_batch, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_route_batch, num_workers=0)
    
    model = RouteGRUModel(
        num_blocks=train_dataset.num_blocks,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_id=train_dataset.pad_id
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct_top1 = 0
        train_correct_top3 = 0
        train_correct_top5 = 0
        train_total = 0
        
        start_time = time.time()
        
        for batch_idx, (hist, curr, targ, next_b, metas) in enumerate(train_loader):
            hist, curr, targ, next_b = hist.to(device), curr.to(device), targ.to(device), next_b.to(device)
            
            optimizer.zero_grad()
            logits = model(hist, curr, targ)
            
            loss = criterion(logits, next_b)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * hist.size(0)
            
            _, top5_preds = logits.topk(5, dim=-1)
            train_correct_top1 += (top5_preds[:, 0] == next_b).sum().item()
            train_correct_top3 += (top5_preds[:, :3] == next_b.unsqueeze(1)).any(dim=-1).sum().item()
            train_correct_top5 += (top5_preds == next_b.unsqueeze(1)).any(dim=-1).sum().item()
            train_total += hist.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        train_loss /= train_total
        train_acc1 = train_correct_top1 / train_total if train_total > 0 else 0.0
        train_acc3 = train_correct_top3 / train_total if train_total > 0 else 0.0
        train_acc5 = train_correct_top5 / train_total if train_total > 0 else 0.0
        
        model.eval()
        val_loss = 0.0
        val_correct_top1 = 0
        val_correct_top3 = 0
        val_correct_top5 = 0
        val_total = 0
        
        with torch.no_grad():
            for hist, curr, targ, next_b, metas in val_loader:
                hist, curr, targ, next_b = hist.to(device), curr.to(device), targ.to(device), next_b.to(device)
                
                logits = model(hist, curr, targ)
                loss = criterion(logits, next_b)
                
                val_loss += loss.item() * hist.size(0)
                
                _, top5_preds = logits.topk(5, dim=-1)
                val_correct_top1 += (top5_preds[:, 0] == next_b).sum().item()
                val_correct_top3 += (top5_preds[:, :3] == next_b.unsqueeze(1)).any(dim=-1).sum().item()
                val_correct_top5 += (top5_preds == next_b.unsqueeze(1)).any(dim=-1).sum().item()
                val_total += hist.size(0)
                
        val_loss /= val_total if val_total > 0 else 1.0
        val_acc1 = val_correct_top1 / val_total if val_total > 0 else 0.0
        val_acc3 = val_correct_top3 / val_total if val_total > 0 else 0.0
        val_acc5 = val_correct_top5 / val_total if val_total > 0 else 0.0
        
        epoch_time = time.time() - start_time
        
        logger.info(f"=== Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) ===")
        logger.info(f"Train - Loss: {train_loss:.4f}, Top-1: {train_acc1:.4f}, Top-3: {train_acc3:.4f}, Top-5: {train_acc5:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Top-1: {val_acc1:.4f}, Top-3: {val_acc3:.4f}, Top-5: {val_acc5:.4f}")
        
        if val_acc1 >= best_val_acc:
            best_val_acc = val_acc1
            
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": "route_gru",
                "history_len": args.history_len,
                "num_blocks": train_dataset.num_blocks,
                "pad_id": train_dataset.pad_id,
                "train_manifest": args.train_manifest,
                "val_manifest": args.val_manifest,
                "metrics": {
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "val_acc3": val_acc3,
                    "val_acc5": val_acc5
                }
            }
            torch.save(checkpoint, save_path)
            logger.info(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    train_route()
