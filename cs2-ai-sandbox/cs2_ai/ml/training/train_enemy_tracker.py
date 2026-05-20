from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from torch.utils.data import DataLoader, Dataset

from cs2_ai.config import MAX_ENEMIES
from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor, build_enemy_position_target
from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available

if torch_available():
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
else:
    torch = None
    F = None
    SummaryWriter = None


@dataclass(slots=True)
class TrackerTrainingBatch:
    features: 'torch.Tensor'
    target_positions: 'torch.Tensor'
    target_confidences: 'torch.Tensor'


class EnemyTrackerSequenceTorchDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.feature_extractor = EnemyTrackerFeatureExtractor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        return self.base_dataset.get_sample_metadata(idx)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence_sample = self.base_dataset[idx]
        features = self.feature_extractor.extract(sequence_sample.sequence)
        sample_metadata = self.base_dataset.get_sample_metadata(idx)
        tick_indices = list(sample_metadata['tick_indices'])
        target_tick = int(sample_metadata['target_tick'])
        target_ticks = tick_indices[1:] + [target_tick]

        target_positions = np.zeros((len(target_ticks), MAX_ENEMIES, 3), dtype=np.float32)
        target_confidences = np.zeros((len(target_ticks), MAX_ENEMIES), dtype=np.float32)

        for t_idx, tick in enumerate(target_ticks):
            target_state = self.base_dataset.build_state_for_sample_tick(sample_metadata, tick)
            target_positions[t_idx] = build_enemy_position_target(target_state)
            for enemy_idx, enemy in enumerate(target_state.enemies[:MAX_ENEMIES]):
                if enemy.is_alive:
                    target_confidences[t_idx, enemy_idx] = 1.0

        return features.astype(np.float32), target_positions, target_confidences


class EnemyTrackerTrainer:
    def __init__(self, model: 'torch.nn.Module', device: str, learning_rate: float, log_interval: int = 100, writer: 'SummaryWriter' | None = None):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log_interval = log_interval
        self.writer = writer
        self.global_step = 0

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
            mask = batch.target_confidences.unsqueeze(-1).expand_as(pred_positions)
            pos_loss = F.mse_loss(pred_positions * mask, batch.target_positions * mask, reduction='sum')
            active_count = torch.clamp(batch.target_confidences.sum(), min=1.0)
            pos_loss = pos_loss / active_count
            conf_loss = F.binary_cross_entropy_with_logits(pred_confidences, batch.target_confidences)
            loss = pos_loss + conf_loss

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.global_step += 1

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_pos_loss += float(pos_loss.item()) * batch_size
            total_conf_loss += float(conf_loss.item()) * batch_size

            if training and batch_idx % self.log_interval == 0:
                print(f'Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}')
                if self.writer:
                    self.writer.add_scalar('Train/Loss_Total', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Loss_Position', pos_loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Loss_Confidence', conf_loss.item(), self.global_step)

        if total_samples == 0:
            return {'loss': 0.0, 'pos_loss': 0.0, 'conf_loss': 0.0}

        return {
            'loss': total_loss / total_samples,
            'pos_loss': total_pos_loss / total_samples,
            'conf_loss': total_conf_loss / total_samples,
        }

    def _to_training_batch(self, batch: tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']) -> TrackerTrainingBatch:
        features, target_pos, target_conf = batch
        return TrackerTrainingBatch(
            features=features.to(self.device, non_blocking=True),
            target_positions=target_pos.to(self.device, non_blocking=True),
            target_confidences=target_conf.to(self.device, non_blocking=True),
        )


def collate_tracker_batch(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
    features = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    target_pos = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    target_conf = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32)
    return features, target_pos, target_conf


def save_checkpoint(
    save_path: Path,
    model: 'torch.nn.Module',
    args: argparse.Namespace,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    dataset_label: str,
    input_dim: int,
    demo_names: list[str],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'enemy_tracker_lstm',
        'input_dim': input_dim,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'dataset_source': dataset_label,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
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
    parser.add_argument('--split-mode', choices=['demo', 'round', 'random'], default='demo')
    parser.add_argument('--alive-only', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-samples-per-demo', type=int, default=None)
    parser.add_argument('--max-cached-demos', type=int, default=2)
    parser.add_argument('--show-index-progress', action='store_true')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'enemy_tracker_bc.pt')
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> EnemyTrackerSequenceTorchDataset:
    base_dataset = MultiDemoSequenceDataset(
        dataset_dir=args.dataset_dir,
        subdir='clean_play_ticks',
        seq_len=args.seq_len,
        stride=args.stride,
        alive_only=args.alive_only,
        max_samples_total=args.max_samples,
        max_samples_per_demo=args.max_samples_per_demo,
        max_cached_demos=args.max_cached_demos,
        show_progress=args.show_index_progress,
    )
    print(f'Demo files indexed: {len(base_dataset.demo_paths)}')
    print(f'Sequence samples built: {len(base_dataset)}')
    return EnemyTrackerSequenceTorchDataset(base_dataset)


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_enemy_tracker.py')
        return 0

    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    try:
        dataset = build_dataset(args)
    except FileNotFoundError as exc:
        print(exc)
        print('No clean_play_ticks parquet found. Run parser/cleaner first.')
        return 1

    dataset_len = len(dataset)
    if dataset_len == 0:
        print('Tracker training dataset is empty.')
        return 1

    train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
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
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = PROJECT_ROOT / 'runs' / f'tracker_{timestamp}'
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter else None
    
    trainer = EnemyTrackerTrainer(model=model, device=device, learning_rate=args.lr, log_interval=args.log_interval, writer=writer)
    demo_names = dataset.base_dataset.get_demo_names()
    dataset_label = str(args.dataset_dir / 'clean_play_ticks')

    print('train_enemy_tracker.py')
    print(f'Device: {device}')
    print(f'Dataset source: {dataset_label}')
    print(f'Demo count: {len(demo_names)}')
    print(f'Total samples: {dataset_len}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Split mode: {args.split_mode}')
    print(f'Feature dim: {feature_extractor.feature_dim()}')
    print(f'Save path: {args.save_path}')

    best_val_loss = math.inf
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
            save_checkpoint(args.save_path, model, args, train_metrics, val_metrics, dataset_label, feature_extractor.feature_dim(), demo_names)
            print(f'  saved checkpoint -> {args.save_path}')
            
        if writer:
            writer.add_scalar('Val/Loss_Total', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Loss_Position', val_metrics['pos_loss'], epoch)
            writer.add_scalar('Val/Loss_Confidence', val_metrics['conf_loss'], epoch)

    print('Training finished.')
    print(f'Best val loss: {best_val_loss:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


