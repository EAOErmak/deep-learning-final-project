from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

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

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from cs2_ai.features.movement_features import MovementFeatureExtractor, build_movement_target
from cs2_ai.ml.models.decision_dqn import DecisionDQN
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available

if torch_available():
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
else:
    torch = None
    F = None
    SummaryWriter = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass(slots=True)
class TrainingBatch:
    features: 'torch.Tensor'
    targets: 'torch.Tensor'


class MovementSequenceTorchDataset(Dataset):
    """Wrap a sequence dataset for supervised movement training."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.feature_extractor = MovementFeatureExtractor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        return self.base_dataset.get_sample_metadata(idx)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        sequence_sample = self.base_dataset[idx]
        features = self.feature_extractor.extract(sequence_sample.sequence)

        sample_metadata = self.base_dataset.get_sample_metadata(idx)
        tick_indices = list(sample_metadata['tick_indices'])
        target_tick = int(sample_metadata['target_tick'])
        target_ticks = tick_indices[1:] + [target_tick]

        target = np.zeros((len(target_ticks), 8), dtype=np.float32)
        for t_idx, tick in enumerate(target_ticks):
            target_state = self.base_dataset.build_state_for_sample_tick(sample_metadata, tick)
            target[t_idx] = build_movement_target(target_state)

        return features.astype(np.float32), target.astype(np.float32)


class MovementTrainer:
    def __init__(
        self,
        model: 'torch.nn.Module',
        device: str,
        learning_rate: float,
        show_batch_progress: bool = True,
        log_every: int = 25,
        writer: 'SummaryWriter' | None = None,
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.show_batch_progress = show_batch_progress
        self.log_every = max(1, log_every)
        self.writer = writer
        self.global_step = 0

    def train_epoch(self, loader: DataLoader, epoch_idx: int, total_epochs: int) -> dict[str, float]:
        self.model.train()
        return self._run_epoch(loader, training=True, phase='train', epoch_idx=epoch_idx, total_epochs=total_epochs)

    def eval_epoch(self, loader: DataLoader, epoch_idx: int, total_epochs: int) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, training=False, phase='val', epoch_idx=epoch_idx, total_epochs=total_epochs)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        phase: str,
        epoch_idx: int,
        total_epochs: int,
    ) -> dict[str, float]:
        total_loss = 0.0
        total_binary_loss = 0.0
        total_move_loss = 0.0
        total_samples = 0

        iterator = loader
        progress = None
        if self.show_batch_progress and tqdm is not None:
            iterator = tqdm(
                loader,
                desc=f'{phase} epoch {epoch_idx}/{total_epochs}',
                leave=False,
                unit='batch',
            )
            progress = iterator

        for batch_idx, batch in enumerate(iterator, start=1):
            batch = self._to_training_batch(batch)
            logits = self.model(batch.features)
            binary_logits = logits[..., :6]
            move_logits = logits[..., 6:8]
            binary_targets = batch.targets[..., :6]
            move_targets = batch.targets[..., 6:8]

            binary_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
            move_loss = F.mse_loss(torch.tanh(move_logits), move_targets)
            loss = binary_loss + move_loss

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.global_step += 1

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_binary_loss += float(binary_loss.item()) * batch_size
            total_move_loss += float(move_loss.item()) * batch_size

            if progress is not None and (batch_idx == 1 or batch_idx % self.log_every == 0):
                progress.set_postfix(
                    loss=f'{(total_loss / total_samples):.4f}',
                    bin=f'{(total_binary_loss / total_samples):.4f}',
                    move=f'{(total_move_loss / total_samples):.4f}',
                )
                if training and self.writer:
                    self.writer.add_scalar('Train/Loss_Total', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Loss_Binary', binary_loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Loss_Move', move_loss.item(), self.global_step)

        if total_samples == 0:
            return {'loss': 0.0, 'binary_loss': 0.0, 'move_loss': 0.0}

        return {
            'loss': total_loss / total_samples,
            'binary_loss': total_binary_loss / total_samples,
            'move_loss': total_move_loss / total_samples,
        }

    def _to_training_batch(self, batch: tuple['torch.Tensor', 'torch.Tensor']) -> TrainingBatch:
        features, targets = batch
        return TrainingBatch(
            features=features.to(self.device, non_blocking=True),
            targets=targets.to(self.device, non_blocking=True),
        )


def collate_movement_batch(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple['torch.Tensor', 'torch.Tensor']:
    features = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    targets = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    return features, targets


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
        'model_type': 'decision_dqn_movement',
        'input_dim': input_dim,
        'action_dim': 8,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'dataset_source': dataset_label,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'feature_order': 'MovementFeatureExtractor sequence output',
    }
    torch.save(checkpoint, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a supervised movement model from clean_play_ticks')
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'dataset')
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--split-mode', choices=['demo', 'round', 'random'], default='demo')
    parser.add_argument('--alive-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-samples-per-demo', type=int, default=None)
    parser.add_argument('--max-cached-demos', type=int, default=2)
    parser.add_argument('--show-index-progress', action='store_true')
    parser.add_argument('--disable-batch-progress', action='store_true')
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'movement_bc.pt')
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> MovementSequenceTorchDataset:
    print('Scanning clean_play_ticks parquet files...')
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
    return MovementSequenceTorchDataset(base_dataset)


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_movement.py')
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
        print('Movement training dataset is empty. Try smaller seq_len/stride or another demo set.')
        return 1

    train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_movement_batch,
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_movement_batch,
        pin_memory=(device == 'cuda'),
    )

    feature_extractor = MovementFeatureExtractor()
    model = DecisionDQN(input_dim=feature_extractor.feature_dim(), action_dim=8, hidden_dim=args.hidden_dim).to(device)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = PROJECT_ROOT / 'runs' / f'movement_{timestamp}'
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter else None
    
    trainer = MovementTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        show_batch_progress=not args.disable_batch_progress,
        log_every=args.log_every,
        writer=writer,
    )

    demo_names = dataset.base_dataset.get_demo_names()
    dataset_label = str(args.dataset_dir / 'clean_play_ticks')

    print('train_movement.py')
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
    best_train_metrics: dict[str, float] = {'loss': math.inf, 'binary_loss': math.inf, 'move_loss': math.inf}
    best_val_metrics: dict[str, float] = {'loss': math.inf, 'binary_loss': math.inf, 'move_loss': math.inf}

    for epoch in range(1, args.epochs + 1):
        print(f'Starting epoch {epoch}/{args.epochs}...')
        train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch, total_epochs=args.epochs)
        val_metrics = trainer.eval_epoch(val_loader, epoch_idx=epoch, total_epochs=args.epochs) if len(val_dataset) > 0 else {'loss': train_metrics['loss'], 'binary_loss': train_metrics['binary_loss'], 'move_loss': train_metrics['move_loss']}
        print(
            f'Epoch {epoch}/{args.epochs} | '
            f'train_loss={train_metrics["loss"]:.4f} '
            f'(bin={train_metrics["binary_loss"]:.4f}, move={train_metrics["move_loss"]:.4f}) | '
            f'val_loss={val_metrics["loss"]:.4f} '
            f'(bin={val_metrics["binary_loss"]:.4f}, move={val_metrics["move_loss"]:.4f})'
        )
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            save_checkpoint(args.save_path, model, args, train_metrics, val_metrics, dataset_label, feature_extractor.feature_dim(), demo_names)
            print(f'  saved checkpoint -> {args.save_path}')
            
        if writer:
            writer.add_scalar('Val/Loss_Total', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Loss_Binary', val_metrics['binary_loss'], epoch)
            writer.add_scalar('Val/Loss_Move', val_metrics['move_loss'], epoch)

    print('Training finished.')
    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Best train metrics: {best_train_metrics}')
    print(f'Best val metrics: {best_val_metrics}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


