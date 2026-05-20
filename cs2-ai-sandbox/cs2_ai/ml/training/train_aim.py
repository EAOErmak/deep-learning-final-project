from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from torch.utils.data import DataLoader, Dataset

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from cs2_ai.features.aim_features import AimFeatureExtractor, build_aim_target
from cs2_ai.features.feature_contract import FeatureSchema
from cs2_ai.ml.models.aim_attention import AimAttentionModel
from cs2_ai.ml.utils.tensorboard_utils import close_summary_writer, create_summary_writer, log_scalar_dict, tensorboard_available
from cs2_ai.ml.utils.torch_utils import get_device, set_seed, torch_available

if torch_available():
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None


@dataclass(slots=True)
class AimTrainingBatch:
    features: 'torch.Tensor'
    targets: 'torch.Tensor'
    visible_enemy_mask: 'torch.Tensor'
    sample_ids: list[str]
    demo_names: list[str]


@dataclass(slots=True)
class AimBaselinePrior:
    mouse_mean: np.ndarray
    fire_prob: float
    rightclick_prob: float


AIM_TARGET_EPS = 1e-6


def get_base_dataset_and_index(dataset: Any, idx: int) -> tuple[Any, int]:
    curr_dataset = dataset
    curr_idx = idx
    while hasattr(curr_dataset, 'dataset') and hasattr(curr_dataset, 'indices'):
        curr_idx = curr_dataset.indices[curr_idx]
        curr_dataset = curr_dataset.dataset
    return curr_dataset, curr_idx


class AimSequenceTorchDataset(Dataset):
    def __init__(self, base_dataset, seq_len: int, require_spotted_enemy: bool = True):
        self.base_dataset = base_dataset
        self.feature_extractor = AimFeatureExtractor(seq_len=seq_len)
        self.require_spotted_enemy = require_spotted_enemy
        self.valid_indices = self._build_valid_indices()

    def __len__(self) -> int:
        return len(self.valid_indices)

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        base_idx = self.valid_indices[idx]
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, base_idx)
        return ds.get_sample_metadata(real_idx)

    def _build_valid_indices(self) -> list[int]:
        if not self.require_spotted_enemy:
            return list(range(len(self.base_dataset)))

        valid_indices: list[int] = []
        for idx in range(len(self.base_dataset)):
            sample_metadata = self.get_base_sample_metadata(idx)
            if self.sample_has_spotted_enemy(sample_metadata):
                valid_indices.append(idx)
        return valid_indices

    def get_base_sample_metadata(self, idx: int) -> dict[str, object]:
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        return ds.get_sample_metadata(real_idx)

    def sample_has_spotted_enemy(self, sample_metadata: dict[str, object]) -> bool:
        ds = self.base_dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        ticks_to_check = [int(tick) for tick in sample_metadata.get('tick_indices', ())]
        ticks_to_check.append(int(sample_metadata['target_tick']))
        for tick in ticks_to_check:
            try:
                state = ds.build_state_for_sample_tick(sample_metadata, tick)
            except Exception:
                continue
            if any(enemy.spotted and enemy.is_alive for enemy in state.enemies):
                return True
        return False

    def build_target(self, idx: int | None = None, sample_metadata: dict[str, object] | None = None) -> np.ndarray:
        if sample_metadata is None:
            if idx is None:
                raise ValueError('Either idx or sample_metadata must be provided')
            base_idx = self.valid_indices[idx]
            ds, real_idx = get_base_dataset_and_index(self.base_dataset, base_idx)
            sample_metadata = ds.get_sample_metadata(real_idx)
        else:
            ds = self.base_dataset
            while hasattr(ds, 'dataset'):
                ds = ds.dataset
        target_tick = int(sample_metadata['target_tick'])
        target_state = ds.build_state_for_sample_tick(sample_metadata, target_tick)
        try:
            next_state = ds.build_state_for_sample_tick(sample_metadata, target_tick + 1)
        except Exception:
            next_state = target_state
        return build_aim_target(target_state, next_state).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        base_idx = self.valid_indices[idx]
        sequence_sample = self.base_dataset[base_idx]
        features = self.feature_extractor.extract(sequence_sample.sequence)
        sample_metadata = self.get_sample_metadata(idx)
        target = self.build_target(sample_metadata=sample_metadata)
        visible_enemy_mask = np.asarray([1.0 if self.sample_has_spotted_enemy(sample_metadata) else 0.0], dtype=np.float32)
        meta = {
            'sample_id': str(sample_metadata['sample_id']),
            'demo_name': str(sample_metadata['demo_name']),
        }
        return features.astype(np.float32), target.astype(np.float32), visible_enemy_mask, meta


class AimTrainer:
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
        total_aim_loss = 0.0
        total_shoot_bce = 0.0
        total_rightclick_bce = 0.0
        total_active_aim_loss = 0.0
        total_active_aim_samples = 0
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
            aim_delta, shoot_logits, rightclick_logits = self.model(batch.features)
            mouse_targets = batch.targets[:, 5:7]
            fire_targets = batch.targets[:, 2].unsqueeze(1)
            rightclick_targets = batch.targets[:, 3].unsqueeze(1)

            aim_loss_raw = F.mse_loss(torch.tanh(aim_delta), mouse_targets, reduction='none')
            shoot_loss_raw = F.binary_cross_entropy_with_logits(shoot_logits, fire_targets, reduction='none')
            rightclick_loss_raw = F.binary_cross_entropy_with_logits(rightclick_logits, rightclick_targets, reduction='none')
            aim_loss_per_sample = aim_loss_raw.mean(dim=1)
            shoot_bce_per_sample = shoot_loss_raw.squeeze(1)
            rightclick_bce_per_sample = rightclick_loss_raw.squeeze(1)
            loss_per_sample = aim_loss_per_sample + shoot_bce_per_sample + rightclick_bce_per_sample
            loss = loss_per_sample.mean()
            active_aim_mask = torch.any(torch.abs(mouse_targets) > AIM_TARGET_EPS, dim=1) & (batch.visible_enemy_mask.squeeze(1) > 0.5)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss_per_sample.sum().item())
            total_aim_loss += float(aim_loss_per_sample.sum().item())
            total_shoot_bce += float(shoot_bce_per_sample.sum().item())
            total_rightclick_bce += float(rightclick_bce_per_sample.sum().item())
            if torch.any(active_aim_mask):
                total_active_aim_loss += float(aim_loss_per_sample[active_aim_mask].sum().item())
                total_active_aim_samples += int(active_aim_mask.sum().item())

            if training and writer is not None:
                global_step = (epoch - 1) * total_batches + batch_idx
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/aim_loss_step', aim_loss_per_sample.mean().item(), global_step)
                writer.add_scalar('train/shoot_bce_step', shoot_bce_per_sample.mean().item(), global_step)
                writer.add_scalar('train/rightclick_bce_step', rightclick_bce_per_sample.mean().item(), global_step)
                writer.add_scalar('train/active_aim_rate_step', active_aim_mask.float().mean().item(), global_step)
                if torch.any(active_aim_mask):
                    writer.add_scalar('train/active_aim_loss_step', aim_loss_per_sample[active_aim_mask].mean().item(), global_step)
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
            return {'loss': 0.0, 'aim_loss': 0.0, 'shoot_bce': 0.0, 'rightclick_bce': 0.0, 'active_aim_loss': 0.0, 'active_aim_rate': 0.0, 'seen_sample_ids': set(), 'per_demo_loss': {}, 'per_demo_seen_counts': {}}

        return {
            'loss': total_loss / total_samples,
            'aim_loss': total_aim_loss / total_samples,
            'shoot_bce': total_shoot_bce / total_samples,
            'rightclick_bce': total_rightclick_bce / total_samples,
            'active_aim_loss': (total_active_aim_loss / total_active_aim_samples) if total_active_aim_samples else 0.0,
            'active_aim_rate': float(total_active_aim_samples / total_samples),
            'seen_sample_ids': seen_sample_ids,
            'per_demo_loss': {demo: per_demo_sum[demo] / per_demo_count[demo] for demo in sorted(per_demo_sum)},
            'per_demo_seen_counts': {demo: len(ids) for demo, ids in sorted(seen_demo_sample_ids.items())},
        }

    def _to_training_batch(self, batch: tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, str]]]) -> AimTrainingBatch:
        features, targets, visible_enemy_mask, metas = batch
        return AimTrainingBatch(
            features=features.to(self.device, non_blocking=True),
            targets=targets.to(self.device, non_blocking=True),
            visible_enemy_mask=visible_enemy_mask.to(self.device, non_blocking=True),
            sample_ids=[str(meta['sample_id']) for meta in metas],
            demo_names=[str(meta['demo_name']) for meta in metas],
        )


def collate_aim_batch(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]]) -> tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, str]]]:
    features = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    targets = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    visible_enemy_mask = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32)
    metas = [item[3] for item in batch]
    return features, targets, visible_enemy_mask, metas


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


def resolve_indexed_dataset(dataset) -> tuple[AimSequenceTorchDataset, list[int]]:
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        return dataset.dataset, [int(idx) for idx in dataset.indices]
    return dataset, list(range(len(dataset)))


def _iter_aim_targets(dataset):
    source_dataset, indices = resolve_indexed_dataset(dataset)
    for idx in indices:
        sample_metadata = source_dataset.get_sample_metadata(idx)
        yield source_dataset.build_target(sample_metadata=sample_metadata)


def _clip_probability(probability: float) -> float:
    return float(np.clip(probability, 1e-6, 1.0 - 1e-6))


def _binary_cross_entropy_from_probability(target: float, probability: float) -> float:
    clipped_probability = _clip_probability(probability)
    return float(-(target * np.log(clipped_probability) + (1.0 - target) * np.log(1.0 - clipped_probability)))


def _is_active_mouse_target(mouse_target: np.ndarray) -> bool:
    return bool(np.any(np.abs(mouse_target) > AIM_TARGET_EPS))


def build_aim_baseline_prior(dataset) -> AimBaselinePrior:
    total_targets = 0
    mouse_sum = np.zeros(2, dtype=np.float64)
    fire_sum = 0.0
    rightclick_sum = 0.0

    for target in _iter_aim_targets(dataset):
        total_targets += 1
        mouse_sum += target[5:7]
        fire_sum += float(target[2])
        rightclick_sum += float(target[3])

    if total_targets == 0:
        return AimBaselinePrior(mouse_mean=np.zeros(2, dtype=np.float32), fire_prob=0.5, rightclick_prob=0.5)

    return AimBaselinePrior(
        mouse_mean=(mouse_sum / total_targets).astype(np.float32),
        fire_prob=float(fire_sum / total_targets),
        rightclick_prob=float(rightclick_sum / total_targets),
    )


def evaluate_aim_baseline(dataset, prior: AimBaselinePrior) -> dict[str, float]:
    total_targets = 0
    total_loss = 0.0
    total_aim_loss = 0.0
    total_shoot_bce = 0.0
    total_rightclick_bce = 0.0
    total_active_aim_loss = 0.0
    total_active_aim_samples = 0

    for target in _iter_aim_targets(dataset):
        mouse_target = target[5:7]
        aim_loss = float(np.mean(np.square(mouse_target - prior.mouse_mean)))
        shoot_bce = _binary_cross_entropy_from_probability(float(target[2]), prior.fire_prob)
        rightclick_bce = _binary_cross_entropy_from_probability(float(target[3]), prior.rightclick_prob)

        total_targets += 1
        total_loss += aim_loss + shoot_bce + rightclick_bce
        total_aim_loss += aim_loss
        total_shoot_bce += shoot_bce
        total_rightclick_bce += rightclick_bce
        if _is_active_mouse_target(mouse_target):
            total_active_aim_samples += 1
            total_active_aim_loss += aim_loss

    if total_targets == 0:
        return {'loss': 0.0, 'aim_loss': 0.0, 'shoot_bce': 0.0, 'rightclick_bce': 0.0, 'active_aim_loss': 0.0, 'active_aim_rate': 0.0}

    return {
        'loss': total_loss / total_targets,
        'aim_loss': total_aim_loss / total_targets,
        'shoot_bce': total_shoot_bce / total_targets,
        'rightclick_bce': total_rightclick_bce / total_targets,
        'active_aim_loss': (total_active_aim_loss / total_active_aim_samples) if total_active_aim_samples else 0.0,
        'active_aim_rate': float(total_active_aim_samples / total_targets),
    }


def build_coverage_summary(metrics: dict[str, object], expected_demo_counts: dict[str, int]) -> dict[str, object]:
    seen_counts = dict(metrics.get('per_demo_seen_counts', {}))
    per_demo = {}
    covered_demos = 0
    for demo_name, expected_count in sorted(expected_demo_counts.items()):
        seen_count = int(seen_counts.get(demo_name, 0))
        coverage = float(seen_count / expected_count) if expected_count else 0.0
        if seen_count >= expected_count:
            covered_demos += 1
        per_demo[demo_name] = {
            'seen': seen_count,
            'expected': int(expected_count),
            'coverage': coverage,
            'avg_loss': float(metrics.get('per_demo_loss', {}).get(demo_name, 0.0)),
        }
    total_expected = int(sum(expected_demo_counts.values()))
    total_seen = int(sum(seen_counts.values()))
    return {
        'total_seen': total_seen,
        'total_expected': total_expected,
        'coverage': float(total_seen / total_expected) if total_expected else 0.0,
        'covered_demos': covered_demos,
        'demo_count': len(expected_demo_counts),
        'per_demo': per_demo,
    }


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
        'model_type': 'aim_attention',
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
    parser = argparse.ArgumentParser(description='Train a supervised aim model from clean_play_ticks')
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
    parser.add_argument('--allow-no-spotted-enemy', action='store_true')
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--tensorboard-run-name', type=str, default=None)
    parser.add_argument('--disable-tensorboard', action='store_true')
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'aim_bc.pt')
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> AimSequenceTorchDataset:
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
    return AimSequenceTorchDataset(base_dataset, seq_len=args.seq_len, require_spotted_enemy=not args.allow_no_spotted_enemy)


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_aim.py')
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
        print('Aim training dataset is empty. Try smaller seq_len/stride or another demo set.')
        return 1

    print('Building train/val split...')
    train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
    train_expected_counts = collect_expected_demo_counts(train_dataset)
    val_expected_counts = collect_expected_demo_counts(val_dataset)
    print('Preparing dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_aim_batch, pin_memory=(device == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_aim_batch, pin_memory=(device == 'cuda'))

    print('Initializing model and trainer...')
    feature_extractor = AimFeatureExtractor(seq_len=args.seq_len)
    feature_schema = feature_extractor.schema()
    model = AimAttentionModel(input_dim=feature_extractor.feature_dim()).to(device)
    trainer = AimTrainer(model=model, device=device, learning_rate=args.lr, log_interval=args.log_interval)
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
            default_prefix='aim',
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

    print('train_aim.py')
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
    print('Computing train-prior baseline metrics...')
    baseline_prior = build_aim_baseline_prior(train_dataset)
    train_baseline = evaluate_aim_baseline(train_dataset, baseline_prior)
    val_baseline = evaluate_aim_baseline(val_dataset, baseline_prior) if len(val_dataset) > 0 else dict(train_baseline)
    print(
        'Baseline | '
        f'train_loss={train_baseline["loss"]:.4f} '
        f'(aim={train_baseline["aim_loss"]:.4f}, shoot_bce={train_baseline["shoot_bce"]:.4f}, active_aim={train_baseline["active_aim_loss"]:.4f}, active_rate={train_baseline["active_aim_rate"]:.2%}) | '
        f'val_loss={val_baseline["loss"]:.4f} '
        f'(aim={val_baseline["aim_loss"]:.4f}, shoot_bce={val_baseline["shoot_bce"]:.4f}, active_aim={val_baseline["active_aim_loss"]:.4f}, active_rate={val_baseline["active_aim_rate"]:.2%})'
    )

    best_val_loss = math.inf
    try:
        for epoch in range(1, args.epochs + 1):
            print(f'Starting epoch {epoch}/{args.epochs}...')
            train_metrics = trainer.train_epoch(train_loader, epoch=epoch, writer=writer)
            val_metrics = trainer.eval_epoch(val_loader, epoch=epoch, writer=writer) if len(val_dataset) > 0 else {'loss': train_metrics['loss'], 'aim_loss': train_metrics['aim_loss'], 'shoot_bce': train_metrics['shoot_bce'], 'rightclick_bce': train_metrics['rightclick_bce'], 'active_aim_loss': train_metrics['active_aim_loss'], 'active_aim_rate': train_metrics['active_aim_rate'], 'seen_sample_ids': set(), 'per_demo_loss': {}, 'per_demo_seen_counts': {}}
            print(f'Epoch {epoch}/{args.epochs} | train_loss={train_metrics["loss"]:.4f} (aim={train_metrics["aim_loss"]:.4f}, shoot_bce={train_metrics["shoot_bce"]:.4f}, rightclick_bce={train_metrics["rightclick_bce"]:.4f}) | val_loss={val_metrics["loss"]:.4f} (aim={val_metrics["aim_loss"]:.4f}, shoot_bce={val_metrics["shoot_bce"]:.4f}, rightclick_bce={val_metrics["rightclick_bce"]:.4f})')
            print(
                'Aim active | '
                f'train_rate={train_metrics["active_aim_rate"]:.2%} '
                f'(loss={train_metrics["active_aim_loss"]:.4f}) | '
                f'val_rate={val_metrics["active_aim_rate"]:.2%} '
                f'(loss={val_metrics["active_aim_loss"]:.4f})'
            )
            print(
                'Baseline   | '
                f'train_loss={train_baseline["loss"]:.4f} '
                f'(aim={train_baseline["aim_loss"]:.4f}, shoot_bce={train_baseline["shoot_bce"]:.4f}) | '
                f'val_loss={val_baseline["loss"]:.4f} '
                f'(aim={val_baseline["aim_loss"]:.4f}, shoot_bce={val_baseline["shoot_bce"]:.4f})'
            )

            train_coverage = build_coverage_summary(train_metrics, train_expected_counts)
            val_coverage = build_coverage_summary(val_metrics, val_expected_counts)
            print_coverage_summary('train', train_coverage)
            if val_expected_counts:
                print_coverage_summary('val', val_coverage)
            append_epoch_summary(epoch_log_path, {'epoch': epoch, 'train': {'loss': train_metrics['loss'], 'aim_loss': train_metrics['aim_loss'], 'shoot_bce': train_metrics['shoot_bce'], 'rightclick_bce': train_metrics['rightclick_bce'], 'active_aim_loss': train_metrics['active_aim_loss'], 'active_aim_rate': train_metrics['active_aim_rate'], 'coverage': train_coverage}, 'val': {'loss': val_metrics['loss'], 'aim_loss': val_metrics['aim_loss'], 'shoot_bce': val_metrics['shoot_bce'], 'rightclick_bce': val_metrics['rightclick_bce'], 'active_aim_loss': val_metrics['active_aim_loss'], 'active_aim_rate': val_metrics['active_aim_rate'], 'coverage': val_coverage}, 'baseline': {'train': train_baseline, 'val': val_baseline}})

            log_scalar_dict(writer, 'train', train_metrics, epoch, ignored_keys={'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'})
            log_scalar_dict(writer, 'val', val_metrics, epoch, ignored_keys={'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'})
            log_scalar_dict(writer, 'baseline/train', train_baseline, epoch)
            log_scalar_dict(writer, 'baseline/val', val_baseline, epoch)
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
