from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from cs2_ai.config import MAX_ENEMIES
from cs2_ai.dataset.parquet_loader import load_first_clean_play_ticks
from cs2_ai.dataset.sequence_dataset import PerspectiveSequenceDataset
from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor, build_enemy_position_target
from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available
from game_state import GameState

if torch_available():
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None


@dataclass(slots=True)
class TrackerTrainingBatch:
    features: "torch.Tensor"
    target_positions: "torch.Tensor"
    target_confidences: "torch.Tensor"


class EnemyTrackerSequenceTorchDataset(Dataset):
    """Wrap PerspectiveSequenceDataset for supervised enemy tracker training.

    Each item returns:
    - features: [seq_len, feature_dim]
    - target_positions: [MAX_ENEMIES, 3]
    - target_confidences: [MAX_ENEMIES]
    """

    def __init__(self, base_dataset: PerspectiveSequenceDataset):
        self.base_dataset = base_dataset
        self.feature_extractor = EnemyTrackerFeatureExtractor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence_sample = self.base_dataset[idx]
        features = self.feature_extractor.extract(sequence_sample.sequence)

        sample_index = self.base_dataset.samples[idx]
        round_number = int(sample_index['round_number'])
        perspective_steamid = int(sample_index['perspective_steamid'])
        tick_indices = list(sample_index['tick_indices'])
        target_tick = int(sample_index['target_tick'])
        
        target_ticks = tick_indices[1:] + [target_tick]
        
        target_positions = np.zeros((len(target_ticks), MAX_ENEMIES, 3), dtype=np.float32)
        target_confidences = np.zeros((len(target_ticks), MAX_ENEMIES), dtype=np.float32)
        
        for t_idx, tick in enumerate(target_ticks):
            target_state = self.base_dataset.game_state_builder.build_from_tick_rows(
                self.base_dataset.round_tick_rows[round_number][tick],
                perspective_steamid,
            )
            target_positions[t_idx] = build_enemy_position_target(target_state)
            for i, enemy in enumerate(target_state.enemies[:MAX_ENEMIES]):
                if enemy.is_alive:
                    target_confidences[t_idx, i] = 1.0

        return features.astype(np.float32), target_positions, target_confidences

class EnemyTrackerTrainer:
    def __init__(self, model: "torch.nn.Module", device: str, learning_rate: float, log_interval: int = 100):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log_interval = log_interval

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        return self._run_epoch(loader, training=True, epoch=epoch)

    def eval_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, training=False, epoch=epoch)

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int) -> dict[str, float]:
        total_loss = 0.0
        total_pos_loss = 0.0
        total_conf_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(loader):
            batch = self._to_training_batch(batch)
            pred_positions, pred_confidences = self.model(batch.features)
            
            # Mask out position loss for enemies that don't exist (confidence target == 0)
            # Confidences are [B, 5] -> expand to [B, 5, 3] for mask
            mask = batch.target_confidences.unsqueeze(-1).expand_as(pred_positions)
            
            # Compute MSE only on valid slots
            pos_loss = F.mse_loss(pred_positions * mask, batch.target_positions * mask, reduction='sum')
            # Normalize by number of active enemies to avoid huge variance (add epsilon to prevent div 0)
            active_count = torch.clamp(batch.target_confidences.sum(), min=1.0)
            pos_loss = pos_loss / active_count
            
            conf_loss = F.binary_cross_entropy_with_logits(pred_confidences, batch.target_confidences)
            
            loss = pos_loss + conf_loss

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_pos_loss += float(pos_loss.item()) * batch_size
            total_conf_loss += float(conf_loss.item()) * batch_size
            
            if training and batch_idx % self.log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        if total_samples == 0:
            return {'loss': 0.0, 'pos_loss': 0.0, 'conf_loss': 0.0}

        return {
            'loss': total_loss / total_samples,
            'pos_loss': total_pos_loss / total_samples,
            'conf_loss': total_conf_loss / total_samples,
        }

    def _to_training_batch(self, batch: tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]) -> TrackerTrainingBatch:
        features, target_pos, target_conf = batch
        return TrackerTrainingBatch(
            features=features.to(self.device, non_blocking=True),
            target_positions=target_pos.to(self.device, non_blocking=True),
            target_confidences=target_conf.to(self.device, non_blocking=True),
        )


def collate_tracker_batch(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    features = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    target_pos = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    target_conf = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32)
    return features, target_pos, target_conf


def split_dataset(dataset: Dataset, val_split: float, seed: int) -> tuple[Dataset, Dataset]:
    total_len = len(dataset)
    if total_len < 2 or val_split <= 0.0:
        return dataset, Subset(dataset, [])
    indices = list(range(total_len))
    random.Random(seed).shuffle(indices)
    val_len = max(1, int(total_len * val_split))
    train_len = max(1, total_len - val_len)
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def save_checkpoint(
    save_path: Path,
    model: "torch.nn.Module",
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    dataset_path: Path,
    input_dim: int,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'enemy_tracker_lstm',
        'input_dim': input_dim,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'dataset_file': str(dataset_path),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }
    torch.save(checkpoint, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train supervised enemy tracker from clean_play_ticks')
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'dataset')
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--alive-only', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'enemy_tracker_bc.pt')
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> tuple[Path, EnemyTrackerSequenceTorchDataset]:
    parquet_path, tick_df = load_first_clean_play_ticks(args.dataset_dir)
    base_dataset = PerspectiveSequenceDataset(
        tick_df=tick_df,
        seq_len=args.seq_len,
        stride=args.stride,
        alive_only=args.alive_only,
    )
    dataset = EnemyTrackerSequenceTorchDataset(base_dataset)
    if args.max_samples is not None:
        capped_len = min(len(dataset), args.max_samples)
        dataset = Subset(dataset, list(range(capped_len)))
    return parquet_path, dataset


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_enemy_tracker.py')
        return 0

    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    try:
        parquet_path, dataset = build_dataset(args)
    except FileNotFoundError as exc:
        print(exc)
        print('No clean_play_ticks parquet found. Run parser/cleaner first.')
        return 1

    dataset_len = len(dataset)
    if dataset_len == 0:
        print('Tracker training dataset is empty.')
        return 1

    train_dataset, val_dataset = split_dataset(dataset, args.val_split, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tracker_batch,
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tracker_batch,
        pin_memory=(device == 'cuda'),
    )

    feature_extractor = EnemyTrackerFeatureExtractor()
    model = EnemyTrackerLSTM(input_dim=feature_extractor.feature_dim(), output_enemies=MAX_ENEMIES).to(device)
    trainer = EnemyTrackerTrainer(model=model, device=device, learning_rate=args.lr, log_interval=args.log_interval)

    print('train_enemy_tracker.py')
    print(f'Device: {device}')
    print(f'Dataset file: {parquet_path}')
    print(f'Total samples: {dataset_len}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Feature dim: {feature_extractor.feature_dim()}')
    print(f'Save path: {args.save_path}')

    best_val_loss = math.inf
    best_train_metrics: dict[str, float] = {'loss': math.inf, 'pos_loss': math.inf, 'conf_loss': math.inf}
    best_val_metrics: dict[str, float] = {'loss': math.inf, 'pos_loss': math.inf, 'conf_loss': math.inf}

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.eval_epoch(val_loader, epoch=epoch) if len(val_dataset) > 0 else {'loss': train_metrics['loss'], 'pos_loss': train_metrics['pos_loss'], 'conf_loss': train_metrics['conf_loss']}
        print(
            f'Epoch {epoch}/{args.epochs} | '
            f'train_loss={train_metrics["loss"]:.4f} '
            f'(pos={train_metrics["pos_loss"]:.4f}, conf={train_metrics["conf_loss"]:.4f}) | '
            f'val_loss={val_metrics["loss"]:.4f} '
            f'(pos={val_metrics["pos_loss"]:.4f}, conf={val_metrics["conf_loss"]:.4f})'
        )
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            save_checkpoint(args.save_path, model, args, train_metrics, val_metrics, parquet_path, feature_extractor.feature_dim())
            print(f'  saved checkpoint -> {args.save_path}')

    print('Training finished.')
    print(f'Best val loss: {best_val_loss:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
