from __future__ import annotations

import argparse
import json
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
from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor, build_enemy_confidence_target, build_enemy_position_target, build_enemy_roster
from cs2_ai.features.feature_contract import FeatureSchema
from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
from cs2_ai.ml.utils.tensorboard_utils import close_summary_writer, create_summary_writer, log_scalar_dict, tensorboard_available
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available

if torch_available():
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None


@dataclass(slots=True)
class TrackerTrainingBatch:
    features: 'torch.Tensor'
    target_positions: 'torch.Tensor'
    target_confidences: 'torch.Tensor'
    sample_ids: list[str]
    demo_names: list[str]


def get_base_dataset_and_index(dataset: Any, idx: int) -> tuple[Any, int]:
    curr_dataset = dataset
    curr_idx = idx
    while hasattr(curr_dataset, 'dataset') and hasattr(curr_dataset, 'indices'):
        curr_idx = curr_dataset.indices[curr_idx]
        curr_dataset = curr_dataset.dataset
    return curr_dataset, curr_idx


class EnemyTrackerSequenceTorchDataset(Dataset):
    def __init__(self, base_dataset, seq_len: int):
        self.base_dataset = base_dataset
        self.feature_extractor = EnemyTrackerFeatureExtractor(seq_len=seq_len)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        return ds.get_sample_metadata(real_idx)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
        sequence_sample = self.base_dataset[idx]
        features = self.feature_extractor.extract(sequence_sample.sequence)
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        sample_metadata = ds.get_sample_metadata(real_idx)
        tick_indices = list(sample_metadata['tick_indices'])
        target_tick = int(sample_metadata['target_tick'])
        target_ticks = tick_indices[1:] + [target_tick]
        roster_steamids = build_enemy_roster(sequence_sample.sequence)

        target_positions = np.zeros((len(target_ticks), MAX_ENEMIES, 3), dtype=np.float32)
        target_confidences = np.zeros((len(target_ticks), MAX_ENEMIES), dtype=np.float32)
        for t_idx, tick in enumerate(target_ticks):
            target_state = ds.build_truth_state_for_sample_tick(sample_metadata, tick)
            target_positions[t_idx] = build_enemy_position_target(target_state, roster_steamids)
            target_confidences[t_idx] = build_enemy_confidence_target(target_state, roster_steamids)

        meta = {
            'sample_id': str(sample_metadata['sample_id']),
            'demo_name': str(sample_metadata['demo_name']),
            'roster_steamids': [int(steamid) for steamid in roster_steamids],
        }
        return features.astype(np.float32), target_positions, target_confidences, meta


class EnemyTrackerTrainer:
    def __init__(self, model: 'torch.nn.Module', device: str, learning_rate: float, log_interval: int = 100):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.log_interval = log_interval

    def train_epoch(self, loader: DataLoader, epoch: int, writer: Any | None = None) -> dict[str, object]:
        self.model.train()
        return self._run_epoch(loader, training=True, epoch=epoch, writer=writer)

    def eval_epoch(self, loader: DataLoader, epoch: int, writer: Any | None = None) -> dict[str, object]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, training=False, epoch=epoch, writer=writer)

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int, writer: Any | None = None) -> dict[str, object]:
        total_loss = 0.0
        total_pos_loss = 0.0
        total_conf_loss = 0.0
        total_samples = 0
        total_batches = len(loader)
        phase_name = 'Train' if training else 'Val'
        per_demo_sum: dict[str, float] = {}
        per_demo_count: dict[str, int] = {}
        seen_sample_ids: set[str] = set()
        seen_demo_sample_ids: dict[str, set[str]] = {}
        first_batch_loaded = False

        print(f'{phase_name} epoch {epoch} | Preparing first batch...')

        for batch_idx, batch in enumerate(loader):
            if not first_batch_loaded:
                print(f'{phase_name} epoch {epoch} | First batch loaded.')
                first_batch_loaded = True
            batch = self._to_training_batch(batch)
            pred_positions, pred_confidences = self.model(batch.features)
            mask = batch.target_confidences.unsqueeze(-1).expand_as(pred_positions)
            pos_loss_raw = F.mse_loss(pred_positions * mask, batch.target_positions * mask, reduction='none')
            active_counts = torch.clamp(batch.target_confidences.sum(dim=(1, 2)), min=1.0)
            pos_loss_per_sample = pos_loss_raw.sum(dim=(1, 2, 3)) / active_counts
            conf_loss_raw = F.binary_cross_entropy_with_logits(pred_confidences, batch.target_confidences, reduction='none')
            conf_loss_per_sample = conf_loss_raw.mean(dim=(1, 2))
            loss_per_sample = pos_loss_per_sample + conf_loss_per_sample
            loss = loss_per_sample.mean()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss_per_sample.sum().item())
            total_pos_loss += float(pos_loss_per_sample.sum().item())
            total_conf_loss += float(conf_loss_per_sample.sum().item())

            if training and writer is not None:
                global_step = (epoch - 1) * total_batches + batch_idx
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/pos_loss_step', pos_loss_per_sample.mean().item(), global_step)
                writer.add_scalar('train/conf_loss_step', conf_loss_per_sample.mean().item(), global_step)
                if batch_idx % 10 == 0:
                    writer.flush()

            for sample_id, demo_name, sample_loss in zip(batch.sample_ids, batch.demo_names, loss_per_sample.detach().cpu().tolist()):
                seen_sample_ids.add(sample_id)
                seen_demo_sample_ids.setdefault(demo_name, set()).add(sample_id)
                per_demo_sum[demo_name] = per_demo_sum.get(demo_name, 0.0) + float(sample_loss)
                per_demo_count[demo_name] = per_demo_count.get(demo_name, 0) + 1

            should_log_batch = (
                batch_idx == 0
                or (batch_idx + 1) % self.log_interval == 0
                or batch_idx == total_batches - 1
            )
            if should_log_batch:
                print(f'{phase_name} epoch {epoch} | Batch {batch_idx + 1}/{total_batches} | Loss: {loss.item():.4f} | Seen: {len(seen_sample_ids)}')

        if total_samples == 0:
            return {'loss': 0.0, 'pos_loss': 0.0, 'conf_loss': 0.0, 'seen_sample_ids': set(), 'per_demo_loss': {}, 'per_demo_seen_counts': {}}

        return {
            'loss': total_loss / total_samples,
            'pos_loss': total_pos_loss / total_samples,
            'conf_loss': total_conf_loss / total_samples,
            'seen_sample_ids': seen_sample_ids,
            'per_demo_loss': {demo: per_demo_sum[demo] / per_demo_count[demo] for demo in sorted(per_demo_sum)},
            'per_demo_seen_counts': {demo: len(ids) for demo, ids in sorted(seen_demo_sample_ids.items())},
        }

    def _to_training_batch(self, batch: tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, str]]]) -> TrackerTrainingBatch:
        features, target_pos, target_conf, metas = batch
        return TrackerTrainingBatch(
            features=features.to(self.device, non_blocking=True),
            target_positions=target_pos.to(self.device, non_blocking=True),
            target_confidences=target_conf.to(self.device, non_blocking=True),
            sample_ids=[str(meta['sample_id']) for meta in metas],
            demo_names=[str(meta['demo_name']) for meta in metas],
        )


def collate_tracker_batch(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]]) -> tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, str]]]:
    features = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    target_pos = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    target_conf = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32)
    metas = [item[3] for item in batch]
    return features, target_pos, target_conf, metas


def collect_expected_demo_counts(dataset) -> dict[str, int]:
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        indices = list(dataset.indices)
        source_dataset = dataset.dataset
    else:
        indices = list(range(len(dataset)))
        source_dataset = dataset
    counts: dict[str, int] = {}
    for idx in indices:
        metadata = source_dataset.get_sample_metadata(int(idx))
        demo_name = str(metadata['demo_name'])
        counts[demo_name] = counts.get(demo_name, 0) + 1
    return counts


def build_coverage_summary(metrics: dict[str, object], expected_demo_counts: dict[str, int]) -> dict[str, object]:
    seen_counts = dict(metrics.get('per_demo_seen_counts', {}))
    per_demo = {}
    covered_demos = 0
    for demo_name, expected_count in sorted(expected_demo_counts.items()):
        seen_count = int(seen_counts.get(demo_name, 0))
        coverage = float(seen_count / expected_count) if expected_count else 0.0
        if seen_count >= expected_count:
            covered_demos += 1
        per_demo[demo_name] = {'seen': seen_count, 'expected': int(expected_count), 'coverage': coverage, 'avg_loss': float(metrics.get('per_demo_loss', {}).get(demo_name, 0.0))}
    total_expected = int(sum(expected_demo_counts.values()))
    total_seen = int(sum(seen_counts.values()))
    return {'total_seen': total_seen, 'total_expected': total_expected, 'coverage': float(total_seen / total_expected) if total_expected else 0.0, 'covered_demos': covered_demos, 'demo_count': len(expected_demo_counts), 'per_demo': per_demo}


def print_coverage_summary(phase: str, coverage_summary: dict[str, object]) -> None:
    print(f"{phase} coverage: {coverage_summary['total_seen']}/{coverage_summary['total_expected']} samples ({coverage_summary['coverage']:.2%}) | demos {coverage_summary['covered_demos']}/{coverage_summary['demo_count']}")
    for demo_name, demo_summary in list(coverage_summary['per_demo'].items())[:10]:
        print(f"  {phase} demo {demo_name}: {demo_summary['seen']}/{demo_summary['expected']} ({demo_summary['coverage']:.2%}) | avg_loss={demo_summary['avg_loss']:.4f}")


def append_epoch_summary(log_path: Path, epoch_summary: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(epoch_summary, ensure_ascii=True) + '\n')


def save_checkpoint(save_path: Path, model: 'torch.nn.Module', args: argparse.Namespace, train_metrics: dict[str, object], val_metrics: dict[str, object], dataset_label: str, schema: FeatureSchema, demo_names: list[str]) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'enemy_tracker_lstm',
        'input_dim': schema.feature_dim,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'feature_schema': schema.to_metadata(),
        'dataset_source': dataset_label,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
        'train_metrics': {k: v for k, v in train_metrics.items() if k not in {'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'}},
        'val_metrics': {k: v for k, v in val_metrics.items() if k not in {'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'}},
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
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--tensorboard-run-name', type=str, default=None)
    parser.add_argument('--disable-tensorboard', action='store_true')
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'enemy_tracker_bc.pt')
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> EnemyTrackerSequenceTorchDataset:
    base_dataset = MultiDemoSequenceDataset(dataset_dir=args.dataset_dir, subdir='clean_play_ticks', seq_len=args.seq_len, stride=args.stride, alive_only=args.alive_only, max_samples_total=args.max_samples, max_samples_per_demo=args.max_samples_per_demo, max_cached_demos=args.max_cached_demos, show_progress=args.show_index_progress)
    print(f'Demo files indexed: {len(base_dataset.demo_paths)}')
    print(f'Sequence samples built: {len(base_dataset)}')
    return EnemyTrackerSequenceTorchDataset(base_dataset, seq_len=args.seq_len)


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_enemy_tracker.py')
        return 0

    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    try:
        print('Building dataset...')
        dataset = build_dataset(args)
    except FileNotFoundError as exc:
        print(exc)
        print('No clean_play_ticks parquet found. Run parser/cleaner first.')
        return 1

    dataset_len = len(dataset)
    if dataset_len == 0:
        print('Tracker training dataset is empty.')
        return 1

    print('Building train/val split...')
    train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
    train_expected_counts = collect_expected_demo_counts(train_dataset)
    val_expected_counts = collect_expected_demo_counts(val_dataset)
    print('Preparing dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_tracker_batch, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_tracker_batch, pin_memory=(device == 'cuda'))

    print('Initializing model and trainer...')
    feature_extractor = EnemyTrackerFeatureExtractor(seq_len=args.seq_len)
    feature_schema = feature_extractor.schema()
    model = EnemyTrackerLSTM(input_dim=feature_extractor.feature_dim(), output_enemies=MAX_ENEMIES).to(device)
    trainer = EnemyTrackerTrainer(model=model, device=device, learning_rate=args.lr, log_interval=args.log_interval)
    demo_names = dataset.base_dataset.get_demo_names()
    dataset_label = str(args.dataset_dir / 'clean_play_ticks')
    epoch_log_path = args.save_path.with_name(f'{args.save_path.stem}_epoch_metrics.jsonl')
    writer = None
    run_dir = None

    if args.disable_tensorboard:
        print('TensorBoard: disabled')
    elif tensorboard_available():
        writer, run_dir = create_summary_writer(
            runs_dir=args.runs_dir,
            run_name=args.tensorboard_run_name,
            default_prefix='enemy_tracker',
            save_path=args.save_path,
            config={
                'args': vars(args),
                'device': device,
                'dataset_source': dataset_label,
                'demo_names': demo_names,
            },
        )
        if run_dir is not None:
            print(f'TensorBoard run: {run_dir}')
    else:
        print('TensorBoard: unavailable (install tensorboard to enable event logging)')

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
    print(f'Epoch log: {epoch_log_path}')

    best_val_loss = math.inf
    try:
        for epoch in range(1, args.epochs + 1):
            print(f'Starting epoch {epoch}/{args.epochs}...')
            train_metrics = trainer.train_epoch(train_loader, epoch=epoch, writer=writer)
            val_metrics = trainer.eval_epoch(val_loader, epoch=epoch, writer=writer) if len(val_dataset) > 0 else {'loss': train_metrics['loss'], 'pos_loss': train_metrics['pos_loss'], 'conf_loss': train_metrics['conf_loss'], 'seen_sample_ids': set(), 'per_demo_loss': {}, 'per_demo_seen_counts': {}}
            print(f'Epoch {epoch}/{args.epochs} | train_loss={train_metrics["loss"]:.4f} (pos={train_metrics["pos_loss"]:.4f}, conf={train_metrics["conf_loss"]:.4f}) | val_loss={val_metrics["loss"]:.4f} (pos={val_metrics["pos_loss"]:.4f}, conf={val_metrics["conf_loss"]:.4f})')

            train_coverage = build_coverage_summary(train_metrics, train_expected_counts)
            val_coverage = build_coverage_summary(val_metrics, val_expected_counts)
            print_coverage_summary('train', train_coverage)
            if val_expected_counts:
                print_coverage_summary('val', val_coverage)
            append_epoch_summary(epoch_log_path, {'epoch': epoch, 'train': {'loss': train_metrics['loss'], 'pos_loss': train_metrics['pos_loss'], 'conf_loss': train_metrics['conf_loss'], 'coverage': train_coverage}, 'val': {'loss': val_metrics['loss'], 'pos_loss': val_metrics['pos_loss'], 'conf_loss': val_metrics['conf_loss'], 'coverage': val_coverage}})

            log_scalar_dict(writer, 'train', train_metrics, epoch, ignored_keys={'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'})
            log_scalar_dict(writer, 'val', val_metrics, epoch, ignored_keys={'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'})
            log_scalar_dict(writer, 'train_coverage', train_coverage, epoch, ignored_keys={'per_demo'})
            log_scalar_dict(writer, 'val_coverage', val_coverage, epoch, ignored_keys={'per_demo'})
            if writer is not None:
                writer.flush()

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(args.save_path, model, args, train_metrics, val_metrics, dataset_label, feature_schema, demo_names)
                print(f'  saved checkpoint -> {args.save_path}')
    finally:
        close_summary_writer(writer)

    print('Training finished.')
    print(f'Best val loss: {best_val_loss:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
