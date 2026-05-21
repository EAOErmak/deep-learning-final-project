from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

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
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from cs2_ai.dataset.prebuilt_split_sequence_dataset import PrebuiltSplitSequenceDataset
from cs2_ai.features.feature_contract import FeatureSchema
from cs2_ai.features.movement_features import (
    GRID_NAVIGATION_FEATURE_NAMES,
    MOVEMENT_FEATURE_MODE_LEGACY,
    MOVEMENT_FEATURE_MODE_SOLO_GRID,
    MOVEMENT_FEATURE_NAMES_SOLO_GRID,
    MOVEMENT_TARGET_MODE_ACTION_CHUNK,
    MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE,
    MovementFeatureExtractor,
    build_movement_action_chunk_target_from_tick_rows,
    build_grid_navigation_feature_frame_from_row,
    build_movement_target_from_tick_rows,
    movement_action_names_for_target_mode,
    normalize_movement_feature_mode,
    normalize_movement_target_mode,
)
from cs2_ai.ml.models.decision_dqn import DecisionDQN
from cs2_ai.ml.models.movement_gru import MovementGRU
from cs2_ai.ml.reporting import build_base_training_report, write_training_report
from cs2_ai.ml.training.training_dataset_utils import (
    add_common_training_data_args,
    build_dataset_label,
    filter_dataset_by_trained_rounds,
    resolve_dataset_root as resolve_shared_dataset_root,
    resolve_run_id,
)
from cs2_ai.ml.training.training_ledger import TrainingRoundLedger, collect_round_usage
from cs2_ai.ml.training.shape_assertions import assert_shape, assert_temporal_features
from cs2_ai.ml.utils.tensorboard_utils import close_summary_writer, create_summary_writer, log_scalar_dict, tensorboard_available
from cs2_ai.ml.utils.torch_utils import build_dataloader_kwargs, configure_torch_runtime, get_device, set_seed, torch_available

if torch_available():
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


MOVEMENT_MODEL_DECISION_DQN = 'decision_dqn'
MOVEMENT_MODEL_GRU = 'movement_gru'


@dataclass(slots=True)
class TrainingBatch:
    features: 'torch.Tensor'
    targets: 'torch.Tensor'
    sample_ids: list[str]
    demo_names: list[str]


def get_base_dataset_and_index(dataset: Any, idx: int) -> tuple[Any, int]:
    curr_dataset = dataset
    curr_idx = idx
    while hasattr(curr_dataset, 'dataset') and hasattr(curr_dataset, 'indices'):
        curr_idx = curr_dataset.indices[curr_idx]
        curr_dataset = curr_dataset.dataset
    return curr_dataset, curr_idx


class MovementSequenceTorchDataset(Dataset):
    """Wrap a sequence dataset for supervised movement training."""

    def __init__(
        self,
        base_dataset,
        *,
        target_mode: str = MOVEMENT_TARGET_MODE_ACTION_CHUNK,
        chunk_len: int = 8,
        use_grid_navigation_features: bool = False,
        movement_feature_mode: str = MOVEMENT_FEATURE_MODE_LEGACY,
    ):
        self.base_dataset = base_dataset
        self.feature_extractor = MovementFeatureExtractor(
            seq_len=getattr(base_dataset, "seq_len", None),
            use_grid_navigation_features=use_grid_navigation_features,
            movement_feature_mode=movement_feature_mode,
        )
        self.target_mode = normalize_movement_target_mode(target_mode)
        self.chunk_len = int(chunk_len)
        self.movement_feature_mode = normalize_movement_feature_mode(movement_feature_mode)
        self.use_grid_navigation_features = self.feature_extractor.requires_grid_navigation_features
        if self.chunk_len <= 0:
            raise ValueError('chunk_len must be positive.')
        self.action_names = movement_action_names_for_target_mode(target_mode)
        self.valid_indices = self._build_valid_indices()

    def __len__(self) -> int:
        return len(self.valid_indices)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def target_len(self) -> int:
        if self.target_mode == MOVEMENT_TARGET_MODE_ACTION_CHUNK:
            return self.chunk_len
        return int(getattr(self.base_dataset, 'seq_len', self.chunk_len))

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        base_idx = self.valid_indices[idx]
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, base_idx)
        return ds.get_sample_metadata(real_idx)

    def _build_valid_indices(self) -> list[int]:
        if self.target_mode == MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE:
            return list(range(len(self.base_dataset)))
        valid_indices: list[int] = []
        for idx in range(len(self.base_dataset)):
            sample_metadata = self._get_base_sample_metadata(idx)
            if self._resolve_target_ticks(sample_metadata) is not None:
                valid_indices.append(idx)
        return valid_indices

    def _get_base_sample_metadata(self, idx: int) -> dict[str, object]:
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        return ds.get_sample_metadata(real_idx)

    def _resolve_round_tick_rows(self, sample_metadata: dict[str, object]) -> dict[int, object]:
        ds = self.base_dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        round_number = int(sample_metadata['round_number'])
        if hasattr(ds, '_get_demo_tick_rows'):
            demo_name = str(sample_metadata['demo_name'])
            return ds._get_demo_tick_rows(demo_name)[round_number]
        if hasattr(ds, 'round_tick_rows'):
            return ds.round_tick_rows[round_number]
        raise ValueError('Unsupported base dataset type for movement target resolution.')

    def _resolve_target_ticks(self, sample_metadata: dict[str, object]) -> list[int] | None:
        if self.target_mode == MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE:
            tick_indices = [int(tick) for tick in sample_metadata['tick_indices']]
            return tick_indices[1:] + [int(sample_metadata['target_tick'])]

        round_tick_rows = self._resolve_round_tick_rows(sample_metadata)
        round_ticks = sorted(int(tick) for tick in round_tick_rows)
        target_tick = int(sample_metadata['target_tick'])
        if target_tick not in round_ticks:
            return None
        target_idx = round_ticks.index(target_tick)
        target_ticks = round_ticks[target_idx:target_idx + self.chunk_len]
        if len(target_ticks) != self.chunk_len:
            return None
        return target_ticks

    def _build_target_for_ticks(self, sample_metadata: dict[str, object], target_ticks: list[int]) -> np.ndarray:
        round_tick_rows = self._resolve_round_tick_rows(sample_metadata)
        perspective_steamid = int(sample_metadata['perspective_steamid'])
        if self.target_mode == MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE:
            target = np.zeros((len(target_ticks), len(self.action_names)), dtype=np.float32)
            for t_idx, tick in enumerate(target_ticks):
                target[t_idx] = build_movement_target_from_tick_rows(round_tick_rows[int(tick)], perspective_steamid)
            return target

        target = np.zeros((len(target_ticks), len(self.action_names)), dtype=np.float32)
        for t_idx, tick in enumerate(target_ticks):
            target[t_idx] = build_movement_action_chunk_target_from_tick_rows(round_tick_rows[int(tick)], perspective_steamid)
        return target

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        base_idx = self.valid_indices[idx]
        sequence_sample = self.base_dataset[base_idx]
        sample_metadata = self.get_sample_metadata(idx)
        grid_navigation_frames = self._build_grid_navigation_frames(sample_metadata) if self.use_grid_navigation_features else None
        features = self.feature_extractor.extract(sequence_sample.sequence, grid_navigation_frames=grid_navigation_frames)
        target_ticks = self._resolve_target_ticks(sample_metadata)
        if target_ticks is None:
            raise ValueError(f'No valid future target chunk for sample {sample_metadata["sample_id"]}.')
        target = self._build_target_for_ticks(sample_metadata, target_ticks)

        assert_shape(features, (len(sample_metadata['tick_indices']), self.feature_extractor.feature_dim()), 'movement sample features')
        assert_shape(target, (len(target_ticks), self.action_dim), 'movement sample targets')
        if np.any((target != 0.0) & (target != 1.0)):
            raise ValueError(f'movement sample targets must be binary 0/1, got sample {sample_metadata["sample_id"]}.')

        meta = {
            'sample_id': str(sample_metadata['sample_id']),
            'demo_name': str(sample_metadata['demo_name']),
        }
        return features.astype(np.float32), target.astype(np.float32), meta

    def _build_grid_navigation_frames(self, sample_metadata: dict[str, object]) -> list[dict[str, float]]:
        round_tick_rows = self._resolve_round_tick_rows(sample_metadata)
        perspective_steamid = int(sample_metadata['perspective_steamid'])
        frames: list[dict[str, float]] = []
        for tick in sample_metadata['tick_indices']:
            tick_rows = round_tick_rows[int(tick)]
            steamids = pd.to_numeric(tick_rows["steamid"], errors="coerce")
            self_rows = tick_rows.loc[steamids == perspective_steamid]
            if self_rows.empty:
                raise ValueError(f'Perspective player {perspective_steamid} not found for navigation features on tick {tick}.')
            frames.append(build_grid_navigation_feature_frame_from_row(self_rows.iloc[0], strict=True))
        return frames


class MovementTrainer:
    def __init__(
        self,
        model: 'torch.nn.Module',
        model_name: str,
        device: str,
        learning_rate: float,
        show_batch_progress: bool = True,
        log_every: int = 25,
        pos_weight: 'torch.Tensor | None' = None,
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.show_batch_progress = show_batch_progress
        self.log_every = max(1, log_every)
        self.pos_weight = pos_weight.to(device) if pos_weight is not None else None

    def train_epoch(self, loader: DataLoader, epoch_idx: int, total_epochs: int, writer: Any | None = None) -> dict[str, object]:
        self.model.train()
        return self._run_epoch(loader, training=True, phase='train', epoch_idx=epoch_idx, total_epochs=total_epochs, writer=writer)

    def eval_epoch(self, loader: DataLoader, epoch_idx: int, total_epochs: int, writer: Any | None = None) -> dict[str, object]:
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(loader, training=False, phase='val', epoch_idx=epoch_idx, total_epochs=total_epochs, writer=writer)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        phase: str,
        epoch_idx: int,
        total_epochs: int,
        writer: Any | None = None,
    ) -> dict[str, object]:
        total_loss = 0.0
        total_binary_loss = 0.0
        total_samples = 0
        total_batches = len(loader)
        action_dim = int(loader.dataset.dataset.action_dim if hasattr(loader.dataset, 'dataset') and hasattr(loader.dataset, 'indices') else loader.dataset.action_dim)
        per_action_loss_sum = np.zeros(action_dim, dtype=np.float64)
        per_action_tp = np.zeros(action_dim, dtype=np.float64)
        per_action_fp = np.zeros(action_dim, dtype=np.float64)
        per_action_fn = np.zeros(action_dim, dtype=np.float64)
        per_action_tn = np.zeros(action_dim, dtype=np.float64)
        per_action_pred_pos = np.zeros(action_dim, dtype=np.float64)
        per_action_target_pos = np.zeros(action_dim, dtype=np.float64)
        exact_chunk_matches = 0.0
        total_steps = 0
        per_demo_sum: dict[str, float] = {}
        per_demo_count: dict[str, int] = {}
        seen_sample_ids: set[str] = set()
        seen_demo_sample_ids: dict[str, set[str]] = {}
        first_batch_loaded = False

        print(f'{phase} epoch {epoch_idx}/{total_epochs} | Preparing first batch...')

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
            if not first_batch_loaded:
                first_batch_loaded = True
                message = f'{phase} epoch {epoch_idx}/{total_epochs} | First batch loaded.'
                if progress is not None:
                    progress.write(message)
                elif self.show_batch_progress:
                    print(message)
            batch = self._to_training_batch(batch)
            binary_logits = self._normalize_logits(self.model(batch.features), batch.targets, batch.features)
            assert_shape(binary_logits, tuple(batch.targets.shape), 'movement logits')
            binary_targets = batch.targets

            binary_loss_raw = F.binary_cross_entropy_with_logits(
                binary_logits,
                binary_targets,
                pos_weight=self.pos_weight,
                reduction='none',
            )
            binary_loss_per_sample = binary_loss_raw.mean(dim=(1, 2))
            loss_per_sample = binary_loss_per_sample
            loss = loss_per_sample.mean()
            per_action_loss = binary_loss_raw.mean(dim=(0, 1))

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss_per_sample.sum().item())
            total_binary_loss += float(binary_loss_per_sample.sum().item())
            per_action_loss_sum += per_action_loss.detach().cpu().numpy()
            pred_binary = (torch.sigmoid(binary_logits) > 0.5).to(dtype=torch.float32)
            target_binary = (binary_targets > 0.5).to(dtype=torch.float32)
            per_action_tp += (pred_binary * target_binary).sum(dim=(0, 1)).detach().cpu().numpy()
            per_action_fp += (pred_binary * (1.0 - target_binary)).sum(dim=(0, 1)).detach().cpu().numpy()
            per_action_fn += ((1.0 - pred_binary) * target_binary).sum(dim=(0, 1)).detach().cpu().numpy()
            per_action_tn += ((1.0 - pred_binary) * (1.0 - target_binary)).sum(dim=(0, 1)).detach().cpu().numpy()
            per_action_pred_pos += pred_binary.sum(dim=(0, 1)).detach().cpu().numpy()
            per_action_target_pos += target_binary.sum(dim=(0, 1)).detach().cpu().numpy()
            exact_chunk_matches += float((pred_binary == target_binary).all(dim=(1, 2)).sum().item())
            total_steps += int(binary_targets.shape[0] * binary_targets.shape[1])

            if training and writer is not None:
                global_step = (epoch_idx - 1) * total_batches + batch_idx - 1
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/binary_loss_step', binary_loss_per_sample.mean().item(), global_step)
                for action_idx, action_loss_value in enumerate(per_action_loss.detach().cpu().tolist()):
                    writer.add_scalar(f'train/per_action_loss_step/action_{action_idx}', float(action_loss_value), global_step)
                if (batch_idx - 1) % 10 == 0:
                    writer.flush()

            per_sample_losses = loss_per_sample.detach().cpu().tolist()
            for sample_id, demo_name, sample_loss in zip(batch.sample_ids, batch.demo_names, per_sample_losses):
                seen_sample_ids.add(sample_id)
                seen_demo_sample_ids.setdefault(demo_name, set()).add(sample_id)
                per_demo_sum[demo_name] = per_demo_sum.get(demo_name, 0.0) + float(sample_loss)
                per_demo_count[demo_name] = per_demo_count.get(demo_name, 0) + 1

            if progress is not None and (batch_idx == 1 or batch_idx % self.log_every == 0):
                progress.set_postfix(
                    loss=f'{(total_loss / total_samples):.4f}',
                    bin=f'{(total_binary_loss / total_samples):.4f}',
                    seen=len(seen_sample_ids),
                )

            should_log_batch = (
                batch_idx == 1
                or batch_idx % self.log_every == 0
                or batch_idx == total_batches
            )
            if should_log_batch:
                message = (
                    f'{phase} epoch {epoch_idx}/{total_epochs} | '
                    f'Batch {batch_idx}/{total_batches} | '
                    f'Loss: {loss.item():.4f} | Seen: {len(seen_sample_ids)}'
                )
                if progress is not None:
                    progress.write(message)
                elif self.show_batch_progress:
                    print(message)

        if total_samples == 0:
            return {
                'loss': 0.0,
                'binary_loss': 0.0,
                'per_action_loss': {},
                'per_action_accuracy': {},
                'per_action_precision': {},
                'per_action_recall': {},
                'per_action_f1': {},
                'per_action_pred_positive_rate': {},
                'per_action_target_positive_rate': {},
                'chunk_exact_match': 0.0,
                'seen_sample_ids': set(),
                'per_demo_loss': {},
                'per_demo_seen_counts': {},
            }

        action_names = self._resolve_action_names(loader.dataset)
        per_action_precision = {
            name: float(per_action_tp[idx] / max(per_action_tp[idx] + per_action_fp[idx], 1.0))
            for idx, name in enumerate(action_names)
        }
        per_action_recall = {
            name: float(per_action_tp[idx] / max(per_action_tp[idx] + per_action_fn[idx], 1.0))
            for idx, name in enumerate(action_names)
        }
        per_action_f1 = {
            name: float(
                (2.0 * per_action_precision[name] * per_action_recall[name])
                / max(per_action_precision[name] + per_action_recall[name], 1e-8)
            )
            for name in action_names
        }
        return {
            'loss': total_loss / total_samples,
            'binary_loss': total_binary_loss / total_samples,
            'per_action_loss': {
                name: float(per_action_loss_sum[idx] / max(total_batches, 1))
                for idx, name in enumerate(action_names)
            },
            'per_action_accuracy': {
                name: float((per_action_tp[idx] + per_action_tn[idx]) / max(per_action_tp[idx] + per_action_fp[idx] + per_action_fn[idx] + per_action_tn[idx], 1.0))
                for idx, name in enumerate(action_names)
            },
            'per_action_precision': per_action_precision,
            'per_action_recall': per_action_recall,
            'per_action_f1': per_action_f1,
            'per_action_pred_positive_rate': {
                name: float(per_action_pred_pos[idx] / max(total_steps, 1))
                for idx, name in enumerate(action_names)
            },
            'per_action_target_positive_rate': {
                name: float(per_action_target_pos[idx] / max(total_steps, 1))
                for idx, name in enumerate(action_names)
            },
            'chunk_exact_match': float(exact_chunk_matches / max(total_samples, 1)),
            'seen_sample_ids': seen_sample_ids,
            'per_demo_loss': {demo: per_demo_sum[demo] / per_demo_count[demo] for demo in sorted(per_demo_sum)},
            'per_demo_seen_counts': {demo: len(ids) for demo, ids in sorted(seen_demo_sample_ids.items())},
        }

    def _to_training_batch(self, batch: tuple['torch.Tensor', 'torch.Tensor', list[dict[str, str]]]) -> TrainingBatch:
        features, targets, metas = batch
        batch_size, seq_len, feature_dim = assert_temporal_features(
            features,
            seq_len=int(features.shape[1]),
            feature_dim=self._model_input_dim(),
            name='movement batch features',
        )
        assert_shape(targets, (batch_size, int(targets.shape[1]), int(targets.shape[2])), 'movement batch targets')
        return TrainingBatch(
            features=features.to(self.device, non_blocking=True),
            targets=targets.to(self.device, non_blocking=True),
            sample_ids=[str(meta['sample_id']) for meta in metas],
            demo_names=[str(meta['demo_name']) for meta in metas],
        )

    def _model_input_dim(self) -> int:
        if self.model_name == MOVEMENT_MODEL_GRU:
            return int(self.model.input_dim)
        return int(self.model.net[0].in_features)

    def _normalize_logits(self, logits: 'torch.Tensor', targets: 'torch.Tensor', features: 'torch.Tensor') -> 'torch.Tensor':
        if self.model_name == MOVEMENT_MODEL_GRU:
            assert_shape(logits, (int(features.shape[0]), int(targets.shape[1]), int(targets.shape[2])), 'movement gru logits')
            return logits
        assert_shape(logits, (int(features.shape[0]), int(features.shape[1]), int(targets.shape[2])), 'movement dqn logits_raw')
        if int(logits.shape[1]) < int(targets.shape[1]):
            raise ValueError(
                f'movement logits seq_len {int(logits.shape[1])} is shorter than target len {int(targets.shape[1])}.'
            )
        return logits[:, -int(targets.shape[1]):, :]

    def _resolve_action_names(self, dataset) -> list[str]:
        source_dataset = dataset.dataset if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices') else dataset
        return list(source_dataset.action_names)


def collate_movement_batch(batch: list[tuple[np.ndarray, np.ndarray, dict[str, str]]]) -> tuple['torch.Tensor', 'torch.Tensor', list[dict[str, str]]]:
    features = torch.from_numpy(np.stack([item[0] for item in batch]).astype(np.float32, copy=False))
    targets = torch.from_numpy(np.stack([item[1] for item in batch]).astype(np.float32, copy=False))
    batch_size, _, _ = assert_temporal_features(
        features,
        seq_len=int(features.shape[1]),
        feature_dim=int(features.shape[2]),
        name='movement collated features',
    )
    assert_shape(targets, (batch_size, int(targets.shape[1]), int(targets.shape[2])), 'movement collated targets')
    metas = [item[2] for item in batch]
    return features, targets, metas


def inspect_movement_batch(
    batch: tuple['torch.Tensor', 'torch.Tensor', list[dict[str, str]]],
    action_names: tuple[str, ...],
    feature_dim: int,
) -> None:
    features, targets, metas = batch
    print('Inspecting first movement batch...')
    print(f'  features shape: {tuple(features.shape)}')
    print(f'  features dtype: {features.dtype}')
    print(f'  targets shape: {tuple(targets.shape)}')
    print(f'  targets dtype: {targets.dtype}')
    print(f'  feature_dim: {feature_dim}')
    print(f'  action_names: {list(action_names)}')
    if metas:
        first_meta = metas[0]
        print(f'  first sample_id: {first_meta.get("sample_id", "")}')
        print(f'  first demo_name: {first_meta.get("demo_name", "")}')
    positive_ratios = targets.to(dtype=torch.float32).mean(dim=(0, 1)).cpu().tolist()
    print('  target positive ratios:')
    for action_name, ratio in zip(action_names, positive_ratios, strict=True):
        print(f'    {action_name}: {float(ratio):.4f}')


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


MOVEMENT_METRIC_DICT_KEYS = {
    'seen_sample_ids',
    'per_demo_loss',
    'per_demo_seen_counts',
    'per_action_loss',
    'per_action_accuracy',
    'per_action_precision',
    'per_action_recall',
    'per_action_f1',
    'per_action_pred_positive_rate',
    'per_action_target_positive_rate',
}


def print_coverage_summary(phase: str, coverage_summary: dict[str, object]) -> None:
    print(
        f"{phase} coverage: {coverage_summary['total_seen']}/{coverage_summary['total_expected']} "
        f"samples ({coverage_summary['coverage']:.2%}) | demos {coverage_summary['covered_demos']}/{coverage_summary['demo_count']}"
    )
    for demo_name, demo_summary in list(coverage_summary['per_demo'].items())[:10]:
        print(
            f"  {phase} demo {demo_name}: {demo_summary['seen']}/{demo_summary['expected']} "
            f"({demo_summary['coverage']:.2%}) | avg_loss={demo_summary['avg_loss']:.4f}"
        )


def append_epoch_summary(log_path: Path, epoch_summary: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(epoch_summary, ensure_ascii=True) + '\n')


def save_checkpoint(
    save_path: Path,
    model: 'torch.nn.Module',
    args: argparse.Namespace,
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    dataset_label: str,
    schema: FeatureSchema,
    demo_names: list[str],
    action_names: tuple[str, ...],
    *,
    round_dataset_format: str | None = None,
    train_round_usage: list[dict[str, object]],
    val_round_usage: list[dict[str, object]],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_type = 'movement_gru_chunk' if args.model == MOVEMENT_MODEL_GRU else 'decision_dqn_movement'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'input_dim': schema.feature_dim,
        'action_dim': len(action_names),
        'action_names': list(action_names),
        'movement_model_name': args.model,
        'movement_feature_mode': args.movement_feature_mode,
        'target_mode': args.target_mode,
        'chunk_len': int(getattr(model, 'chunk_len', args.chunk_len if args.target_mode == MOVEMENT_TARGET_MODE_ACTION_CHUNK else args.seq_len)),
        'seq_len': args.seq_len,
        'stride': args.stride,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'gru_num_layers': args.num_layers,
        'gru_dropout': args.dropout,
        'feature_schema': schema.to_metadata(),
        'dataset_source': dataset_label,
        'dataset_dir': str(resolve_shared_dataset_root(args, PROJECT_ROOT)),
        'dataset_subdir': args.dataset_subdir,
        'round_dataset_format': round_dataset_format,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
        'train_metrics': {k: v for k, v in train_metrics.items() if k not in MOVEMENT_METRIC_DICT_KEYS},
        'val_metrics': {k: v for k, v in val_metrics.items() if k not in MOVEMENT_METRIC_DICT_KEYS},
        'feature_order': list(schema.feature_names),
        'train_round_count': len(train_round_usage),
        'val_round_count': len(val_round_usage),
        'train_round_uids': [item['round_uid'] for item in train_round_usage],
        'val_round_uids': [item['round_uid'] for item in val_round_usage],
        'rounds_ledger_path': str(args.rounds_ledger_path),
    }
    torch.save(checkpoint, save_path)


def load_checkpoint_if_available(model: 'torch.nn.Module', resume_from: Path | None, device: str) -> bool:
    if resume_from is None or not resume_from.exists():
        return False
    checkpoint = torch.load(resume_from, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Resumed model weights from: {resume_from}')
    return True


def parse_pos_weight_values(raw_value: str | None, action_dim: int) -> np.ndarray | None:
    if raw_value is None:
        return None
    values = [float(part.strip()) for part in str(raw_value).split(',') if part.strip()]
    if len(values) != action_dim:
        raise ValueError(f'Expected {action_dim} comma-separated pos weights, got {len(values)}.')
    return np.asarray(values, dtype=np.float32)


def compute_action_ratios(dataset: MovementSequenceTorchDataset) -> dict[str, object]:
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        source_dataset = dataset.dataset
        indices = [int(idx) for idx in dataset.indices]
    else:
        source_dataset = dataset
        indices = list(range(len(dataset)))
    positives = np.zeros(source_dataset.action_dim, dtype=np.float64)
    total_steps = 0
    for idx in indices:
        sample_metadata = source_dataset.get_sample_metadata(idx)
        target_ticks = source_dataset._resolve_target_ticks(sample_metadata)
        if target_ticks is None:
            continue
        targets = source_dataset._build_target_for_ticks(sample_metadata, target_ticks)
        positives += targets.sum(axis=0)
        total_steps += int(targets.shape[0])
    ratios = positives / max(total_steps, 1)
    return {
        'action_names': list(source_dataset.action_names),
        'positive_counts': positives.astype(np.int64).tolist(),
        'positive_ratios': ratios.astype(np.float64).tolist(),
        'total_steps': int(total_steps),
    }


def compute_pos_weight(dataset: MovementSequenceTorchDataset, mode: str, explicit_values: str | None) -> np.ndarray | None:
    action_dim = dataset.dataset.action_dim if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices') else dataset.action_dim
    explicit = parse_pos_weight_values(explicit_values, action_dim)
    if explicit is not None:
        return explicit
    if mode == 'none':
        return None
    stats = compute_action_ratios(dataset)
    ratios = np.asarray(stats['positive_ratios'], dtype=np.float64)
    pos_weight = np.ones(action_dim, dtype=np.float32)
    for idx, ratio in enumerate(ratios):
        ratio = float(np.clip(ratio, 1e-4, 1.0 - 1e-4))
        pos_weight[idx] = float(np.clip((1.0 - ratio) / ratio, 1.0, 25.0))
    return pos_weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a supervised movement model from clean_play_ticks')
    add_common_training_data_args(parser, project_root=PROJECT_ROOT, legacy_dataset_dir=True)
    parser.add_argument('--trainset-dir', type=Path, default=None)
    parser.add_argument('--train-data', type=Path, default=None)
    parser.add_argument('--val-data', type=Path, default=None)
    parser.add_argument('--test-data', type=Path, default=None)
    parser.add_argument('--model', choices=[MOVEMENT_MODEL_DECISION_DQN, MOVEMENT_MODEL_GRU], default=MOVEMENT_MODEL_GRU)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--target-mode', choices=[MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE, 'next_tick_sequence', MOVEMENT_TARGET_MODE_ACTION_CHUNK], default=MOVEMENT_TARGET_MODE_ACTION_CHUNK)
    parser.add_argument('--chunk-len', type=int, default=8)
    parser.add_argument('--movement-feature-mode', choices=[MOVEMENT_FEATURE_MODE_LEGACY, MOVEMENT_FEATURE_MODE_SOLO_GRID], default=MOVEMENT_FEATURE_MODE_LEGACY)
    parser.add_argument('--use-grid-navigation-features', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gru-num-layers', dest='num_layers', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--gru-dropout', dest='dropout', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--split-mode', choices=['demo', 'round', 'random'], default='demo')
    parser.add_argument('--alive-only', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-samples-per-demo', type=int, default=None)
    parser.add_argument('--max-cached-demos', type=int, default=2)
    parser.add_argument('--show-index-progress', action='store_true')
    parser.add_argument('--disable-batch-progress', action='store_true')
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--movement-pos-weight-mode', choices=['auto', 'none'], default='auto')
    parser.add_argument('--pos-weight-mode', dest='movement_pos_weight_mode', choices=['auto', 'none'], help=argparse.SUPPRESS)
    parser.add_argument('--pos-weight-values', type=str, default=None)
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--tensorboard-run-name', type=str, default=None)
    parser.add_argument('--disable-tensorboard', action='store_true')
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'movement_bc.pt')
    parser.add_argument('--resume-from', type=Path, default=None)
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    return resolve_shared_dataset_root(args, PROJECT_ROOT)


def build_dataset(args: argparse.Namespace) -> MovementSequenceTorchDataset:
    dataset_root = resolve_dataset_root(args)
    print(f'Scanning parquet files in {dataset_root / args.dataset_subdir}...')
    base_dataset = MultiDemoSequenceDataset(
        dataset_dir=dataset_root,
        subdir=args.dataset_subdir,
        seq_len=args.seq_len,
        stride=args.stride,
        alive_only=args.alive_only,
        max_samples_total=args.max_samples,
        max_samples_per_demo=args.max_samples_per_demo,
        max_cached_demos=args.max_cached_demos,
        show_progress=args.show_index_progress,
    )
    print(f'Demo files indexed: {len(base_dataset.demo_paths)}')
    print(f'Base sequence samples built: {len(base_dataset)}')
    dataset = MovementSequenceTorchDataset(
        base_dataset,
        target_mode=normalize_movement_target_mode(args.target_mode),
        chunk_len=args.chunk_len,
        use_grid_navigation_features=args.use_grid_navigation_features,
        movement_feature_mode=args.movement_feature_mode,
    )
    print(f'Filtered movement samples: {len(dataset)}')
    return dataset


def build_prebuilt_split_dataset(parquet_path: Path, args: argparse.Namespace) -> MovementSequenceTorchDataset:
    base_dataset = PrebuiltSplitSequenceDataset(
        parquet_path=parquet_path,
        seq_len=args.seq_len,
        stride=args.stride,
        alive_only=args.alive_only,
        max_samples=args.max_samples,
        show_progress=args.show_index_progress,
    )
    return MovementSequenceTorchDataset(
        base_dataset,
        target_mode=normalize_movement_target_mode(args.target_mode),
        chunk_len=args.chunk_len,
        use_grid_navigation_features=args.use_grid_navigation_features,
        movement_feature_mode=args.movement_feature_mode,
    )


def resolve_prebuilt_split_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    if args.train_data is not None:
        return args.train_data, args.val_data, args.test_data
    if args.trainset_dir is None:
        raise ValueError('trainset_dir or train_data must be provided for prebuilt split loading.')
    train_path = args.trainset_dir / 'train.parquet'
    val_path = args.trainset_dir / 'val.parquet'
    test_path = args.trainset_dir / 'test.parquet'
    return train_path, val_path if val_path.exists() else None, test_path if test_path.exists() else None


def validate_feature_mode(feature_extractor: MovementFeatureExtractor) -> None:
    feature_names = feature_extractor.feature_names()
    if feature_extractor.movement_feature_mode != MOVEMENT_FEATURE_MODE_SOLO_GRID:
        return
    if feature_extractor.feature_dim() != len(MOVEMENT_FEATURE_NAMES_SOLO_GRID):
        raise ValueError(
            f'solo_grid feature_dim mismatch: expected {len(MOVEMENT_FEATURE_NAMES_SOLO_GRID)}, '
            f'got {feature_extractor.feature_dim()}.'
        )
    if any(name.startswith('teammate_') for name in feature_names):
        raise ValueError(f'solo_grid must not include teammate features: {feature_names}')
    required = {
        'next_cell_rel_x',
        'next_cell_rel_y',
        'next_cell_rel_z',
        'next_cell_distance',
        'dwell_pass_through',
        'dwell_short_hold',
        'dwell_medium_hold',
        'dwell_long_hold',
    }
    missing = sorted(required - set(feature_names))
    if missing:
        raise ValueError(f'solo_grid is missing required navigation features: {missing}')


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_movement.py')
        return 0

    args = parse_args()
    args.target_mode = normalize_movement_target_mode(args.target_mode)
    args.movement_feature_mode = normalize_movement_feature_mode(args.movement_feature_mode)
    if args.movement_feature_mode == MOVEMENT_FEATURE_MODE_SOLO_GRID:
        args.use_grid_navigation_features = True
    if args.model == MOVEMENT_MODEL_DECISION_DQN:
        print('WARNING: DecisionDQN movement mode is legacy. Prefer --model movement_gru --target-mode action_chunk.')
    elif args.model == MOVEMENT_MODEL_GRU and args.target_mode == MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE:
        print('WARNING: MovementGRU is recommended with action_chunk target mode. Current target_mode=next_tick_sequence.')
    set_seed(args.seed)
    device = get_device()
    runtime_info = configure_torch_runtime(device)

    try:
        print('Building dataset...')
        using_prebuilt_trainset = args.trainset_dir is not None or args.train_data is not None
        if using_prebuilt_trainset:
            train_path, val_path, test_path = resolve_prebuilt_split_paths(args)
            dataset = build_prebuilt_split_dataset(train_path, args)
        else:
            dataset = build_dataset(args)
    except FileNotFoundError as exc:
        print(exc)
        print(f'No parquet files found under {resolve_dataset_root(args) / args.dataset_subdir}. Run parser/cleaner first.')
        return 1

    dataset_len = len(dataset)
    if dataset_len == 0:
        print('Movement training dataset is empty. Try smaller seq_len/stride or another demo set.')
        return 1
    if args.skip_trained_rounds:
        dataset, skip_info = filter_dataset_by_trained_rounds(
            dataset,
            ledger_path=args.rounds_ledger_path,
            module_name='movement',
            model_name=args.model,
            checkpoint_path=str(args.save_path),
            match_mode=args.ledger_match_mode,
        )
        print(f'total rounds before skip: {skip_info["total_rounds_before_skip"]}')
        print(f'skipped rounds count: {skip_info["skipped_rounds_count"]}')
        print(f'remaining rounds count: {skip_info["remaining_rounds_count"]}')
        if len(dataset) == 0:
            print('Dataset is empty after --skip-trained-rounds filtering.')
            return 1

    print('Building train/val split...')
    using_prebuilt_trainset = args.trainset_dir is not None or args.train_data is not None
    test_dataset = None
    if using_prebuilt_trainset:
        train_dataset = dataset
        _, val_path, test_path = resolve_prebuilt_split_paths(args)
        val_dataset = build_prebuilt_split_dataset(val_path, args) if val_path is not None and val_path.exists() else split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)[1]
        test_dataset = build_prebuilt_split_dataset(test_path, args) if test_path is not None and test_path.exists() else None
    else:
        train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
    train_round_usage = collect_round_usage(train_dataset)
    val_round_usage = collect_round_usage(val_dataset)
    ledger = TrainingRoundLedger.load(args.rounds_ledger_path)
    run_id = resolve_run_id(args, 'movement')
    dataset_root = resolve_dataset_root(args)
    ledger.append_run_rounds(
        run_id=run_id,
        module_name='movement',
        model_name=args.model,
        checkpoint_path=str(args.save_path),
        dataset_dir=str(dataset_root),
        dataset_subdir=args.dataset_subdir,
        split_mode=args.split_mode,
        split='train',
        round_usage=train_round_usage,
    )
    ledger.append_run_rounds(
        run_id=run_id,
        module_name='movement',
        model_name=args.model,
        checkpoint_path=str(args.save_path),
        dataset_dir=str(dataset_root),
        dataset_subdir=args.dataset_subdir,
        split_mode=args.split_mode,
        split='val',
        round_usage=val_round_usage,
    )
    train_expected_counts = collect_expected_demo_counts(train_dataset)
    val_expected_counts = collect_expected_demo_counts(val_dataset)
    print('Computing movement target statistics...')
    dataset_stats = compute_action_ratios(dataset)
    train_pos_weight_np = compute_pos_weight(train_dataset, args.movement_pos_weight_mode, args.pos_weight_values) if len(train_dataset) > 0 else None
    print(f'Target mode: {args.target_mode}')
    print(f'Chunk len: {dataset.target_len}')
    print(f'Target shape: [batch, {dataset.target_len}, {dataset.action_dim}]')
    print(f'Movement feature mode: {args.movement_feature_mode}')
    if dataset.feature_extractor.requires_grid_navigation_features:
        print(f'Grid navigation features enabled: {list(GRID_NAVIGATION_FEATURE_NAMES)}')
    print('Action positive ratios:')
    for name, ratio in zip(dataset.action_names, dataset_stats['positive_ratios'], strict=True):
        print(f'  {name}: {float(ratio):.4f}')
    if train_pos_weight_np is not None:
        print('Pos weights:')
        for name, value in zip(dataset.action_names, train_pos_weight_np.tolist(), strict=True):
            print(f'  {name}: {float(value):.4f}')

    print('Preparing dataloaders...')
    train_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=True)
    val_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_movement_batch,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_movement_batch,
        **val_loader_kwargs,
    )

    print('Initializing model and trainer...')
    feature_extractor = MovementFeatureExtractor(
        seq_len=args.seq_len,
        use_grid_navigation_features=args.use_grid_navigation_features,
        movement_feature_mode=args.movement_feature_mode,
    )
    validate_feature_mode(feature_extractor)
    first_batch = next(iter(train_loader))
    inspect_movement_batch(first_batch, dataset.action_names, feature_extractor.feature_dim())
    expected_target_len = args.chunk_len if args.target_mode == MOVEMENT_TARGET_MODE_ACTION_CHUNK else args.seq_len
    expected_action_dim = 7 if args.target_mode == MOVEMENT_TARGET_MODE_ACTION_CHUNK else 6
    assert_temporal_features(
        first_batch[0],
        seq_len=args.seq_len,
        feature_dim=feature_extractor.feature_dim(),
        name='first movement batch features',
    )
    assert_shape(
        first_batch[1],
        (int(first_batch[1].shape[0]), expected_target_len, expected_action_dim),
        'first movement batch targets',
    )
    feature_schema = feature_extractor.schema()
    model = build_model(
        model_name=args.model,
        input_dim=feature_extractor.feature_dim(),
        action_dim=dataset.action_dim,
        hidden_dim=args.hidden_dim,
        target_len=dataset.target_len,
        gru_num_layers=args.num_layers,
        gru_dropout=args.dropout,
        device=device,
    )
    load_checkpoint_if_available(model, args.resume_from, device)
    trainer = MovementTrainer(
        model=model,
        model_name=args.model,
        device=device,
        learning_rate=args.lr,
        show_batch_progress=not args.disable_batch_progress,
        log_every=args.log_every,
        pos_weight=torch.tensor(train_pos_weight_np, dtype=torch.float32) if train_pos_weight_np is not None else None,
    )

    demo_names = dataset.base_dataset.get_demo_names()
    dataset_label = str(args.trainset_dir) if using_prebuilt_trainset and args.trainset_dir is not None else build_dataset_label(args, PROJECT_ROOT)
    epoch_log_path = args.save_path.with_name(f'{args.save_path.stem}_epoch_metrics.jsonl')
    writer = None
    run_dir = None

    if args.disable_tensorboard:
        print('TensorBoard: disabled')
    elif tensorboard_available():
        writer, run_dir = create_summary_writer(
            runs_dir=args.runs_dir,
            run_name=args.tensorboard_run_name,
            default_prefix='movement',
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

    print('train_movement.py')
    print(f'Device: {device}')
    print(f'Dataset source: {dataset_label}')
    print(f'Demo count: {len(demo_names)}')
    print(f'Total samples: {dataset_len}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    if test_dataset is not None:
        print(f'Test samples: {len(test_dataset)}')
    print(f'DataLoader workers: train={train_loader_kwargs["num_workers"]} val={val_loader_kwargs["num_workers"]}')
    print(f'CUDA tuning: matmul={runtime_info["matmul_precision"]} cudnn_benchmark={runtime_info["cudnn_benchmark"]} tf32={runtime_info["tf32"]}')
    print(f'Split mode: {args.split_mode}')
    print(f'Model: {args.model}')
    print(f'Movement feature dim: {feature_extractor.feature_dim()}')
    print('Movement features:')
    for idx, feature_name in enumerate(feature_extractor.feature_names()):
        print(f'{idx} {feature_name}')
    print(f'Action dim: {dataset.action_dim}')
    print(f'Save path: {args.save_path}')
    print(f'Epoch log: {epoch_log_path}')

    best_val_loss = math.inf
    best_train_metrics: dict[str, object] = {'loss': math.inf, 'binary_loss': math.inf}
    best_val_metrics: dict[str, object] = {'loss': math.inf, 'binary_loss': math.inf}

    try:
        for epoch in range(1, args.epochs + 1):
            print(f'Starting epoch {epoch}/{args.epochs}...')
            train_metrics = trainer.train_epoch(train_loader, epoch_idx=epoch, total_epochs=args.epochs, writer=writer)
            val_metrics = trainer.eval_epoch(val_loader, epoch_idx=epoch, total_epochs=args.epochs, writer=writer) if len(val_dataset) > 0 else {
                'loss': train_metrics['loss'],
                'binary_loss': train_metrics['binary_loss'],
                'per_action_loss': dict(train_metrics['per_action_loss']),
                'per_action_accuracy': dict(train_metrics['per_action_accuracy']),
                'per_action_precision': dict(train_metrics['per_action_precision']),
                'per_action_recall': dict(train_metrics['per_action_recall']),
                'per_action_f1': dict(train_metrics['per_action_f1']),
                'per_action_pred_positive_rate': dict(train_metrics['per_action_pred_positive_rate']),
                'per_action_target_positive_rate': dict(train_metrics['per_action_target_positive_rate']),
                'chunk_exact_match': float(train_metrics['chunk_exact_match']),
                'seen_sample_ids': set(),
                'per_demo_loss': {},
                'per_demo_seen_counts': {},
            }
            print(
                f'Epoch {epoch}/{args.epochs} | '
                f'train_loss={train_metrics["loss"]:.4f} '
                f'(bin={train_metrics["binary_loss"]:.4f}) | '
                f'val_loss={val_metrics["loss"]:.4f} '
                f'(bin={val_metrics["binary_loss"]:.4f})'
            )
            print('Per-action metrics:')
            for action_name in dataset.action_names:
                print(
                    f'  {action_name}: '
                    f'loss={train_metrics["per_action_loss"].get(action_name, 0.0):.4f} '
                    f'prec={train_metrics["per_action_precision"].get(action_name, 0.0):.4f} '
                    f'recall={train_metrics["per_action_recall"].get(action_name, 0.0):.4f} '
                    f'f1={train_metrics["per_action_f1"].get(action_name, 0.0):.4f} '
                    f'pred_pos={train_metrics["per_action_pred_positive_rate"].get(action_name, 0.0):.4f} '
                    f'tgt_pos={train_metrics["per_action_target_positive_rate"].get(action_name, 0.0):.4f}'
                )
            print(f'Chunk exact match: train={train_metrics["chunk_exact_match"]:.4f} val={val_metrics["chunk_exact_match"]:.4f}')

            train_coverage = build_coverage_summary(train_metrics, train_expected_counts)
            val_coverage = build_coverage_summary(val_metrics, val_expected_counts)
            print_coverage_summary('train', train_coverage)
            if val_expected_counts:
                print_coverage_summary('val', val_coverage)

            append_epoch_summary(
                epoch_log_path,
                {
                    'epoch': epoch,
                    'train': {
                        'loss': train_metrics['loss'],
                        'binary_loss': train_metrics['binary_loss'],
                        'per_action_loss': train_metrics['per_action_loss'],
                        'per_action_accuracy': train_metrics['per_action_accuracy'],
                        'per_action_precision': train_metrics['per_action_precision'],
                        'per_action_recall': train_metrics['per_action_recall'],
                        'per_action_f1': train_metrics['per_action_f1'],
                        'per_action_pred_positive_rate': train_metrics['per_action_pred_positive_rate'],
                        'per_action_target_positive_rate': train_metrics['per_action_target_positive_rate'],
                        'chunk_exact_match': train_metrics['chunk_exact_match'],
                        'coverage': train_coverage,
                    },
                    'val': {
                        'loss': val_metrics['loss'],
                        'binary_loss': val_metrics['binary_loss'],
                        'per_action_loss': val_metrics['per_action_loss'],
                        'per_action_accuracy': val_metrics['per_action_accuracy'],
                        'per_action_precision': val_metrics['per_action_precision'],
                        'per_action_recall': val_metrics['per_action_recall'],
                        'per_action_f1': val_metrics['per_action_f1'],
                        'per_action_pred_positive_rate': val_metrics['per_action_pred_positive_rate'],
                        'per_action_target_positive_rate': val_metrics['per_action_target_positive_rate'],
                        'chunk_exact_match': val_metrics['chunk_exact_match'],
                        'coverage': val_coverage,
                    },
                },
            )

            log_scalar_dict(writer, 'train', train_metrics, epoch, ignored_keys=MOVEMENT_METRIC_DICT_KEYS)
            log_scalar_dict(writer, 'val', val_metrics, epoch, ignored_keys=MOVEMENT_METRIC_DICT_KEYS)
            log_scalar_dict(writer, 'train_coverage', train_coverage, epoch, ignored_keys={'per_demo'})
            log_scalar_dict(writer, 'val_coverage', val_coverage, epoch, ignored_keys={'per_demo'})
            if writer is not None:
                writer.flush()

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_train_metrics = train_metrics
                best_val_metrics = val_metrics
                save_checkpoint(
                    args.save_path,
                    model,
                    args,
                    train_metrics,
                    val_metrics,
                    dataset_label,
                    feature_schema,
                    demo_names,
                    dataset.action_names,
                    round_dataset_format=getattr(dataset.base_dataset, 'dataset_format', 'prebuilt') if hasattr(dataset, 'base_dataset') else None,
                    train_round_usage=train_round_usage,
                    val_round_usage=val_round_usage,
                )
                print(f'  saved checkpoint -> {args.save_path}')
    finally:
        close_summary_writer(writer)

    print('Training finished.')
    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Best train metrics: {{"loss": {best_train_metrics["loss"]:.4f}, "binary_loss": {best_train_metrics["binary_loss"]:.4f}}}')
    print(f'Best val metrics: {{"loss": {best_val_metrics["loss"]:.4f}, "binary_loss": {best_val_metrics["binary_loss"]:.4f}}}')
    report = build_base_training_report(
        module_name='movement',
        model_name=args.model,
        dataset_path=dataset_label,
        split_mode=args.split_mode,
        seq_len=args.seq_len,
        chunk_len=dataset.target_len,
        feature_dim=feature_extractor.feature_dim(),
        target_shape=f'[batch, {dataset.target_len}, {dataset.action_dim}]',
        checkpoint_path=str(args.save_path),
        config=vars(args),
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
    )
    report_paths = write_training_report(report)
    print(f'Reports: {report_paths["json"]} | {report_paths["csv"]} | {report_paths["markdown"]}')
    return 0


def build_model(
    *,
    model_name: str,
    input_dim: int,
    action_dim: int,
    hidden_dim: int,
    target_len: int,
    gru_num_layers: int,
    gru_dropout: float,
    device: str,
):
    if model_name == MOVEMENT_MODEL_GRU:
        return MovementGRU(
            input_dim=input_dim,
            action_dim=action_dim,
            chunk_len=target_len,
            hidden_dim=hidden_dim,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
        ).to(device)
    return DecisionDQN(input_dim=input_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)


if __name__ == '__main__':
    raise SystemExit(main())
