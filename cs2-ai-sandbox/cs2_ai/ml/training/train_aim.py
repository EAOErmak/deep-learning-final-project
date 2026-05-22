from __future__ import annotations

import argparse
import json
import math
import random
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
from cs2_ai.dataset.parquet_loader import list_parquet_files, load_parquet, parquet_demo_name
from cs2_ai.dataset.round_identity import make_round_uid
from cs2_ai.features.aim_features import (
    AIM_BINARY_ACTION_NAMES,
    AIM_FEATURE_MODE_DEMO_PROJECTED,
    AIM_FEATURE_MODE_VISION_LIKE,
    AimFeatureExtractor,
    AimStructuredTarget,
    build_aim_structured_target,
    build_demo_projected_vision_target,
)
from cs2_ai.features.feature_contract import FeatureSchema
from cs2_ai.ml.models.aim_attention import AIM_HEAD_MODE_LEGACY, AIM_HEAD_MODE_MULTI_HEAD, AimAttentionModel
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
from cs2_ai.schemas.game_state import GameStateSequence, StateBundle
from cs2_ai.schemas.training_samples import SequenceSample
from cs2_ai.state.game_state_builder import GameStateBuilder

if torch_available():
    import torch
    import torch.nn.functional as F
else:
    torch = None
    F = None


AIM_METRIC_DICT_KEYS = {
    'seen_sample_ids',
    'per_demo_loss',
    'per_demo_seen_counts',
    'fire_precision',
    'fire_recall',
    'fire_f1',
}


@dataclass(slots=True)
class AimTrainingBatch:
    features: 'torch.Tensor'
    aim_delta_targets: 'torch.Tensor'
    binary_action_targets: 'torch.Tensor'
    valid_aim_mask: 'torch.Tensor'
    sample_ids: list[str]
    demo_names: list[str]


ROUND_COLUMN_CANDIDATES = ['round_number', 'total_rounds_played']


@dataclass(frozen=True, slots=True)
class SingleRoundSampleRef:
    sample_id: str
    demo_name: str
    demo_dir: str
    parquet_path: str
    round_number: int
    round_file: str
    round_uid: str
    dataset_subdir: str
    perspective_steamid: int
    tick_indices: tuple[int, ...]
    target_tick: int


class SingleRoundSequenceDataset(Dataset):
    def __init__(
        self,
        round_file: Path,
        *,
        dataset_subdir: str,
        seq_len: int,
        stride: int,
        alive_only: bool,
    ):
        self.round_file = Path(round_file)
        self.dataset_subdir = dataset_subdir
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.alive_only = bool(alive_only)
        self.demo_name = parquet_demo_name(self.round_file, dataset_subdir)
        self.demo_dir = self.round_file.parent.parent.name
        self.game_state_builder = GameStateBuilder()
        self.samples: list[SingleRoundSampleRef] = []
        self._bundle_cache: dict[tuple[int, int], StateBundle] = {}
        self.round_tick_rows = self._load_round_tick_rows()
        self.round_number = next(iter(self.round_tick_rows.keys()), None)
        self._build_index()

    @staticmethod
    def build_sample_id(
        demo_name: str,
        round_number: int,
        perspective_steamid: int,
        tick_indices: tuple[int, ...],
        target_tick: int,
    ) -> str:
        start_tick = int(tick_indices[0]) if tick_indices else -1
        return f'{demo_name}::r{round_number}::p{perspective_steamid}::s{start_tick}::t{target_tick}'

    def _resolve_round_column(self, columns: list[str] | pd.Index) -> str | None:
        for column in ROUND_COLUMN_CANDIDATES:
            if column in columns:
                return column
        return None

    def _parse_round_number_from_path(self) -> int:
        stem = self.round_file.stem
        if stem.startswith('round_'):
            suffix = stem[len('round_'):]
            return int(suffix)
        raise ValueError(f'Cannot infer round number from filename: {self.round_file.name}')

    def _load_round_tick_rows(self) -> dict[int, dict[int, pd.DataFrame]]:
        tick_df = load_parquet(self.round_file)
        round_column = self._resolve_round_column(tick_df.columns)
        if round_column is None:
            tick_df = tick_df.copy()
            tick_df['round_number'] = self._parse_round_number_from_path()
            round_column = 'round_number'
        round_tick_rows: dict[int, dict[int, pd.DataFrame]] = {}
        for round_number, round_df in tick_df.groupby(round_column, sort=True):
            round_number_int = int(round_number)
            round_tick_rows[round_number_int] = {
                int(tick): rows.copy()
                for tick, rows in round_df.groupby('tick', sort=True)
            }
        return round_tick_rows

    def _build_tick_summary(self, round_df: pd.DataFrame) -> pd.DataFrame:
        summary_rows: list[dict[str, object]] = []
        for tick, tick_df in round_df.groupby('tick', sort=True):
            steamids = pd.to_numeric(tick_df['steamid'], errors='coerce').dropna().astype('int64')
            all_ids = sorted(set(int(value) for value in steamids.tolist()))
            if 'is_alive' in tick_df.columns:
                alive_mask = tick_df['is_alive'].fillna(False).astype(bool)
                alive_ids_series = pd.to_numeric(tick_df.loc[alive_mask, 'steamid'], errors='coerce').dropna().astype('int64')
                alive_ids = sorted(set(int(value) for value in alive_ids_series.tolist()))
            else:
                alive_ids = list(all_ids)
            summary_rows.append({'tick': int(tick), 'all_ids': all_ids, 'alive_ids': alive_ids})
        return pd.DataFrame(summary_rows)

    def _build_index(self) -> None:
        if not self.round_tick_rows:
            return
        round_number, tick_rows = next(iter(self.round_tick_rows.items()))
        full_round_df = pd.concat(list(tick_rows.values()), ignore_index=True)
        tick_summary = self._build_tick_summary(full_round_df)
        ticks = tick_summary['tick'].tolist()
        if len(ticks) <= self.seq_len:
            return
        tick_all_ids = {int(row.tick): set(row.all_ids) for row in tick_summary.itertuples(index=False)}
        tick_alive_ids = {int(row.tick): set(row.alive_ids) for row in tick_summary.itertuples(index=False)}
        round_uid = make_round_uid(self.demo_dir, int(round_number), self.round_file.name)
        for start_idx in range(0, len(ticks) - self.seq_len, self.stride):
            seq_ticks = tuple(int(tick) for tick in ticks[start_idx:start_idx + self.seq_len])
            target_tick = int(ticks[start_idx + self.seq_len])
            available_ids = set(tick_alive_ids[target_tick]) if self.alive_only else set(tick_all_ids[target_tick])
            if not available_ids:
                continue
            for tick in seq_ticks:
                available_ids &= tick_all_ids[tick]
                if not available_ids:
                    break
            if not available_ids:
                continue
            for steamid in sorted(available_ids):
                sample_id = self.build_sample_id(
                    demo_name=self.demo_name,
                    round_number=int(round_number),
                    perspective_steamid=int(steamid),
                    tick_indices=seq_ticks,
                    target_tick=target_tick,
                )
                self.samples.append(
                    SingleRoundSampleRef(
                        sample_id=sample_id,
                        demo_name=self.demo_name,
                        demo_dir=self.demo_dir,
                        parquet_path=str(self.round_file),
                        round_number=int(round_number),
                        round_file=self.round_file.name,
                        round_uid=round_uid,
                        dataset_subdir=self.dataset_subdir,
                        perspective_steamid=int(steamid),
                        tick_indices=seq_ticks,
                        target_tick=target_tick,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def get_demo_names(self) -> list[str]:
        return [self.demo_name]

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        return {
            'sample_id': sample.sample_id,
            'demo_name': sample.demo_name,
            'demo_dir': sample.demo_dir,
            'parquet_path': sample.parquet_path,
            'source_file': sample.parquet_path,
            'round_file': sample.round_file,
            'source_round_file': sample.round_file,
            'round_uid': sample.round_uid,
            'round_number': sample.round_number,
            'dataset_subdir': sample.dataset_subdir,
            'perspective_steamid': sample.perspective_steamid,
            'tick_indices': sample.tick_indices,
            'target_tick': sample.target_tick,
        }

    def build_bundle_for_sample_tick(self, sample_metadata: dict[str, object], tick: int) -> StateBundle:
        round_number = int(sample_metadata['round_number'])
        perspective_steamid = int(sample_metadata['perspective_steamid'])
        tick_value = int(tick)
        cache_key = (tick_value, perspective_steamid)
        cached_bundle = self._bundle_cache.get(cache_key)
        if cached_bundle is not None:
            return cached_bundle
        bundle = self.game_state_builder.build_state_bundle_from_tick_rows(
            self.round_tick_rows[round_number][tick_value],
            perspective_steamid,
        )
        self._bundle_cache[cache_key] = bundle
        return bundle

    def build_state_for_sample_tick(self, sample_metadata: dict[str, object], tick: int):
        return self.build_bundle_for_sample_tick(sample_metadata, tick).observed_state

    def __getitem__(self, idx: int) -> SequenceSample:
        sample_index = self.get_sample_metadata(idx)
        round_number = int(sample_index['round_number'])
        perspective_steamid = int(sample_index['perspective_steamid'])
        tick_indices = list(sample_index['tick_indices'])
        target_tick = int(sample_index['target_tick'])
        states = [self.build_state_for_sample_tick(sample_index, int(tick)) for tick in tick_indices]
        target_state = self.build_state_for_sample_tick(sample_index, target_tick)
        sequence = GameStateSequence(perspective_steamid=perspective_steamid, states=states)
        return SequenceSample(
            perspective_steamid=perspective_steamid,
            start_tick=states[0].tick,
            end_tick=states[-1].tick,
            round_number=round_number,
            sequence=sequence,
            target_input=target_state.self_input,
        )


def get_base_dataset_and_index(dataset: Any, idx: int) -> tuple[Any, int]:
    curr_dataset = dataset
    curr_idx = idx
    while hasattr(curr_dataset, 'dataset') and hasattr(curr_dataset, 'indices'):
        curr_idx = curr_dataset.indices[curr_idx]
        curr_dataset = curr_dataset.dataset
    return curr_dataset, curr_idx


class AimSequenceTorchDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        seq_len: int,
        require_spotted_enemy: bool = True,
        feature_mode: str = AIM_FEATURE_MODE_DEMO_PROJECTED,
        vision_dropout_prob: float = 0.15,
        vision_noise_std: float = 0.03,
        vision_confidence_jitter: float = 0.1,
    ):
        self.base_dataset = base_dataset
        self.feature_extractor = AimFeatureExtractor(seq_len=seq_len, feature_mode=feature_mode)
        self.require_spotted_enemy = require_spotted_enemy
        self.feature_mode = feature_mode
        self.vision_dropout_prob = float(max(0.0, min(1.0, vision_dropout_prob)))
        self.vision_noise_std = float(max(0.0, vision_noise_std))
        self.vision_confidence_jitter = float(max(0.0, vision_confidence_jitter))
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
        cache_path = self._valid_indices_cache_path()
        if cache_path is not None and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding='utf-8'))
                valid_indices = [int(idx) for idx in payload.get('valid_indices', [])]
                print(f'Loaded aim relevance cache: {cache_path} ({len(valid_indices)} samples)')
                return valid_indices
            except Exception as exc:
                print(f'WARNING: failed to load aim relevance cache {cache_path}: {exc}')
        valid_indices: list[int] = []
        for idx in range(len(self.base_dataset)):
            sample_metadata = self.get_base_sample_metadata(idx)
            if self.sample_has_spotted_enemy(sample_metadata):
                valid_indices.append(idx)
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({'valid_indices': valid_indices}, ensure_ascii=True), encoding='utf-8')
            except Exception as exc:
                print(f'WARNING: failed to save aim relevance cache {cache_path}: {exc}')
        return valid_indices

    def _valid_indices_cache_path(self) -> Path | None:
        dataset = self.base_dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        dataset_dir = getattr(dataset, 'dataset_dir', None)
        index_cache_key = getattr(dataset, 'index_cache_key', None)
        subdir = getattr(dataset, 'subdir', None)
        if dataset_dir is None or index_cache_key is None or subdir is None:
            return None
        root = Path(dataset_dir) / '.cache' / 'aim_relevance'
        return root / f'{subdir}_spotted_seq_{index_cache_key}.json'

    def get_base_sample_metadata(self, idx: int) -> dict[str, object]:
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        return ds.get_sample_metadata(real_idx)

    def sample_has_spotted_enemy(self, sample_metadata: dict[str, object]) -> bool:
        if self._sample_has_spotted_enemy_fast(sample_metadata):
            return True
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

    def _sample_has_spotted_enemy_fast(self, sample_metadata: dict[str, object]) -> bool:
        ds = self.base_dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        if not hasattr(ds, '_get_demo_tick_rows'):
            return False
        demo_name = str(sample_metadata['demo_name'])
        round_number = int(sample_metadata['round_number'])
        perspective_steamid = int(sample_metadata['perspective_steamid'])
        ticks_to_check = [int(tick) for tick in sample_metadata.get('tick_indices', ())]
        ticks_to_check.append(int(sample_metadata['target_tick']))
        try:
            round_tick_rows = ds._get_demo_tick_rows(demo_name)[round_number]
        except Exception:
            return False
        for tick in ticks_to_check:
            tick_rows = round_tick_rows.get(int(tick))
            if tick_rows is None or tick_rows.empty:
                continue
            if self._tick_rows_have_spotted_enemy(tick_rows, perspective_steamid):
                return True
        return False

    @staticmethod
    def _tick_rows_have_spotted_enemy(tick_rows, perspective_steamid: int) -> bool:
        if 'steamid' not in tick_rows.columns:
            return False
        spotted_column = next((column for column in ('spotted', 'is_spotted') if column in tick_rows.columns), None)
        if spotted_column is None:
            return False
        steamids = np.asarray(np.nan_to_num(np.asarray(pd.to_numeric(tick_rows['steamid'], errors='coerce')), nan=-1), dtype=np.int64)
        self_mask = steamids == int(perspective_steamid)
        if not np.any(self_mask):
            return False
        enemy_mask = steamids != int(perspective_steamid)
        if 'team_num' in tick_rows.columns:
            team_values = pd.to_numeric(tick_rows['team_num'], errors='coerce')
            self_team = team_values.loc[self_mask]
            if not self_team.empty and not pd.isna(self_team.iloc[0]):
                enemy_mask = team_values.ne(self_team.iloc[0]).fillna(False).to_numpy(dtype=bool, copy=False)
        spotted_values = pd.to_numeric(tick_rows[spotted_column], errors='coerce').fillna(0).to_numpy(dtype=float, copy=False) > 0
        if 'is_alive' in tick_rows.columns:
            alive_values = tick_rows['is_alive'].fillna(False).astype(bool).to_numpy(dtype=bool, copy=False)
        else:
            alive_values = np.ones(len(tick_rows), dtype=bool)
        return bool(np.any(enemy_mask & spotted_values & alive_values))

    def build_target(self, idx: int | None = None, sample_metadata: dict[str, object] | None = None) -> AimStructuredTarget:
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
        return build_aim_structured_target(target_state, next_state)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
        base_idx = self.valid_indices[idx]
        sequence_sample = self.base_dataset[base_idx]
        sample_metadata = self.get_sample_metadata(idx)
        vision_target = self._build_training_vision_target(sequence_sample.sequence)
        features = self.feature_extractor.extract(sequence_sample.sequence, vision_target=vision_target)
        target = self.build_target(sample_metadata=sample_metadata)
        assert_shape(features, (len(sample_metadata['tick_indices']), self.feature_extractor.feature_dim()), 'aim sample features')
        assert_shape(target.aim_delta, (4,), 'aim sample aim_delta')
        assert_shape(target.binary_actions, (3,), 'aim sample binary_actions')
        assert_shape(target.valid_aim_mask, (1,), 'aim sample valid_aim_mask')
        meta = {
            'sample_id': str(sample_metadata['sample_id']),
            'demo_name': str(sample_metadata['demo_name']),
        }
        return (
            features.astype(np.float32),
            target.aim_delta.astype(np.float32),
            target.binary_actions.astype(np.float32),
            target.valid_aim_mask.astype(np.float32),
            meta,
        )

    def _build_training_vision_target(self, sequence) -> object | None:
        if self.feature_mode != AIM_FEATURE_MODE_VISION_LIKE:
            return None
        synthetic_target = build_demo_projected_vision_target(sequence.states[-1])
        if synthetic_target is None:
            return None
        if np.random.random() < self.vision_dropout_prob:
            return None
        synthetic_target.screen_dx = float(synthetic_target.screen_dx + np.random.normal(0.0, self.vision_noise_std))
        synthetic_target.screen_dy = float(synthetic_target.screen_dy + np.random.normal(0.0, self.vision_noise_std))
        synthetic_target.confidence = float(np.clip(synthetic_target.confidence + np.random.uniform(-self.vision_confidence_jitter, self.vision_confidence_jitter), 0.0, 1.0))
        return synthetic_target


class AimTrainer:
    def __init__(
        self,
        model: 'torch.nn.Module',
        device: str,
        learning_rate: float,
        head_mode: str,
        aim_weight: float = 1.0,
        binary_weight: float = 1.0,
        confidence_weight: float = 0.25,
        log_interval: int = 100,
        binary_pos_weight: 'torch.Tensor | None' = None,
    ):
        self.model = model
        self.device = device
        self.head_mode = head_mode
        self.aim_weight = float(aim_weight)
        self.binary_weight = float(binary_weight)
        self.confidence_weight = float(confidence_weight)
        self.log_interval = log_interval
        self.binary_pos_weight = binary_pos_weight.to(device) if binary_pos_weight is not None else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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
        total_binary_loss = 0.0
        total_confidence_loss = 0.0
        total_samples = 0
        total_batches = len(loader)
        phase_name = 'Train' if training else 'Val'
        per_demo_sum: dict[str, float] = {}
        per_demo_count: dict[str, int] = {}
        seen_sample_ids: set[str] = set()
        seen_demo_sample_ids: dict[str, set[str]] = {}
        first_batch_loaded = False
        fire_tp = 0.0
        fire_fp = 0.0
        fire_fn = 0.0
        pred_fire_sum = 0.0
        target_fire_sum = 0.0
        valid_sum = 0.0
        mouse_abs_sum = np.zeros(2, dtype=np.float64)
        angle_abs_sum = np.zeros(2, dtype=np.float64)

        print(f'{phase_name} epoch {epoch} | Preparing first batch...')

        for batch_idx, batch in enumerate(loader):
            if not first_batch_loaded:
                print(f'{phase_name} epoch {epoch} | First batch loaded.')
                first_batch_loaded = True
            batch = self._to_training_batch(batch)
            outputs = self.model(batch.features)
            regression_pred, binary_logits, confidence_logits = self._normalize_model_outputs(outputs)

            regression_targets = batch.aim_delta_targets[:, 2:4] if self.head_mode == AIM_HEAD_MODE_LEGACY else batch.aim_delta_targets
            regression_mask = batch.valid_aim_mask.expand(-1, regression_targets.shape[1])
            aim_loss_raw = F.smooth_l1_loss(regression_pred, regression_targets, reduction='none')
            aim_loss_per_sample = self._masked_mean(aim_loss_raw, regression_mask)

            binary_loss_raw = F.binary_cross_entropy_with_logits(
                binary_logits,
                batch.binary_action_targets[:, :binary_logits.shape[1]],
                reduction='none',
                pos_weight=self.binary_pos_weight[:binary_logits.shape[1]] if self.binary_pos_weight is not None else None,
            )
            binary_loss_per_sample = binary_loss_raw.mean(dim=1)

            confidence_loss_per_sample = torch.zeros_like(aim_loss_per_sample)
            if confidence_logits is not None:
                confidence_loss_raw = F.binary_cross_entropy_with_logits(confidence_logits, batch.valid_aim_mask, reduction='none')
                confidence_loss_per_sample = confidence_loss_raw.squeeze(1)

            loss_per_sample = (
                aim_loss_per_sample * self.aim_weight
                + binary_loss_per_sample * self.binary_weight
                + confidence_loss_per_sample * self.confidence_weight
            )
            loss = loss_per_sample.mean()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            batch_size = batch.features.size(0)
            total_samples += batch_size
            total_loss += float(loss_per_sample.sum().item())
            total_aim_loss += float(aim_loss_per_sample.sum().item())
            total_binary_loss += float(binary_loss_per_sample.sum().item())
            total_confidence_loss += float(confidence_loss_per_sample.sum().item())

            fire_probs = torch.sigmoid(binary_logits[:, 0])
            fire_pred = (fire_probs > 0.5).to(dtype=batch.binary_action_targets.dtype)
            fire_target = batch.binary_action_targets[:, 0]
            fire_tp += float((fire_pred * fire_target).sum().item())
            fire_fp += float((fire_pred * (1.0 - fire_target)).sum().item())
            fire_fn += float(((1.0 - fire_pred) * fire_target).sum().item())
            pred_fire_sum += float(fire_pred.sum().item())
            target_fire_sum += float(fire_target.sum().item())

            valid_mask_bool = batch.valid_aim_mask.squeeze(1) > 0.5
            valid_sum += float(valid_mask_bool.sum().item())
            if torch.any(valid_mask_bool):
                if self.head_mode == AIM_HEAD_MODE_MULTI_HEAD:
                    mouse_abs_sum += np.abs(
                        (regression_pred[valid_mask_bool, 2:4] - batch.aim_delta_targets[valid_mask_bool, 2:4]).detach().cpu().numpy()
                    ).sum(axis=0)
                    angle_abs_sum += np.abs(
                        (regression_pred[valid_mask_bool, 0:2] - batch.aim_delta_targets[valid_mask_bool, 0:2]).detach().cpu().numpy()
                    ).sum(axis=0)
                else:
                    mouse_abs_sum += np.abs(
                        (regression_pred[valid_mask_bool, 0:2] - batch.aim_delta_targets[valid_mask_bool, 2:4]).detach().cpu().numpy()
                    ).sum(axis=0)

            if training and writer is not None:
                global_step = (epoch - 1) * total_batches + batch_idx
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/aim_loss_step', aim_loss_per_sample.mean().item(), global_step)
                writer.add_scalar('train/binary_loss_step', binary_loss_per_sample.mean().item(), global_step)
                writer.add_scalar('train/confidence_loss_step', confidence_loss_per_sample.mean().item(), global_step)
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
            return self._empty_metrics()

        fire_precision = float(fire_tp / max(fire_tp + fire_fp, 1.0))
        fire_recall = float(fire_tp / max(fire_tp + fire_fn, 1.0))
        fire_f1 = float((2.0 * fire_precision * fire_recall) / max(fire_precision + fire_recall, 1e-8))
        metric_count = max(valid_sum, 1.0)
        return {
            'loss': total_loss / total_samples,
            'aim_loss': total_aim_loss / total_samples,
            'binary_loss': total_binary_loss / total_samples,
            'confidence_loss': total_confidence_loss / total_samples,
            'mouse_dx_mae': float(mouse_abs_sum[0] / metric_count),
            'mouse_dy_mae': float(mouse_abs_sum[1] / metric_count),
            'mean_angular_error': float(angle_abs_sum.mean() / metric_count) if self.head_mode == AIM_HEAD_MODE_MULTI_HEAD else 0.0,
            'fire_precision': {'fire': fire_precision},
            'fire_recall': {'fire': fire_recall},
            'fire_f1': {'fire': fire_f1},
            'predicted_fire_rate': float(pred_fire_sum / total_samples),
            'target_fire_rate': float(target_fire_sum / total_samples),
            'valid_aim_rate': float(valid_sum / total_samples),
            'seen_sample_ids': seen_sample_ids,
            'per_demo_loss': {demo: per_demo_sum[demo] / per_demo_count[demo] for demo in sorted(per_demo_sum)},
            'per_demo_seen_counts': {demo: len(ids) for demo, ids in sorted(seen_demo_sample_ids.items())},
        }

    def _empty_metrics(self) -> dict[str, object]:
        return {
            'loss': 0.0,
            'aim_loss': 0.0,
            'binary_loss': 0.0,
            'confidence_loss': 0.0,
            'mouse_dx_mae': 0.0,
            'mouse_dy_mae': 0.0,
            'mean_angular_error': 0.0,
            'fire_precision': {'fire': 0.0},
            'fire_recall': {'fire': 0.0},
            'fire_f1': {'fire': 0.0},
            'predicted_fire_rate': 0.0,
            'target_fire_rate': 0.0,
            'valid_aim_rate': 0.0,
            'seen_sample_ids': set(),
            'per_demo_loss': {},
            'per_demo_seen_counts': {},
        }

    def _normalize_model_outputs(self, outputs):
        if self.head_mode == AIM_HEAD_MODE_MULTI_HEAD:
            aim_delta, binary_logits, confidence_logits = outputs
            assert_shape(aim_delta, (None, 4), 'aim multi_head regression output')
            assert_shape(binary_logits, (None, 3), 'aim multi_head binary output')
            assert_shape(confidence_logits, (None, 1), 'aim multi_head confidence output')
            return aim_delta, binary_logits, confidence_logits
        aim_delta, shoot_logits, rightclick_logits = outputs
        binary_logits = torch.cat([shoot_logits, rightclick_logits], dim=1)
        assert_shape(aim_delta, (None, 2), 'aim legacy regression output')
        assert_shape(binary_logits, (None, 2), 'aim legacy binary output')
        return aim_delta, binary_logits, None

    def _masked_mean(self, values: 'torch.Tensor', mask: 'torch.Tensor') -> 'torch.Tensor':
        weighted = values * mask
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return weighted.sum(dim=1) / denom

    def _to_training_batch(
        self,
        batch: tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, str]]],
    ) -> AimTrainingBatch:
        features, aim_delta_targets, binary_action_targets, valid_aim_mask, metas = batch
        feature_shape = assert_temporal_features(
            features,
            seq_len=int(features.shape[1]),
            feature_dim=self.model.input_dim,
            name='aim batch features',
        )
        assert_shape(aim_delta_targets, (feature_shape[0], 4), 'aim batch aim_delta_targets')
        assert_shape(binary_action_targets, (feature_shape[0], 3), 'aim batch binary_action_targets')
        assert_shape(valid_aim_mask, (feature_shape[0], 1), 'aim batch valid_aim_mask')
        return AimTrainingBatch(
            features=features.to(self.device, non_blocking=True),
            aim_delta_targets=aim_delta_targets.to(self.device, non_blocking=True),
            binary_action_targets=binary_action_targets.to(self.device, non_blocking=True),
            valid_aim_mask=valid_aim_mask.to(self.device, non_blocking=True),
            sample_ids=[str(meta['sample_id']) for meta in metas],
            demo_names=[str(meta['demo_name']) for meta in metas],
        )


def collate_aim_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]]
) -> tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, str]]]:
    features = torch.from_numpy(np.stack([item[0] for item in batch]).astype(np.float32, copy=False))
    aim_delta_targets = torch.from_numpy(np.stack([item[1] for item in batch]).astype(np.float32, copy=False))
    binary_action_targets = torch.from_numpy(np.stack([item[2] for item in batch]).astype(np.float32, copy=False))
    valid_aim_mask = torch.from_numpy(np.stack([item[3] for item in batch]).astype(np.float32, copy=False))
    batch_size, _, _ = assert_temporal_features(features, seq_len=int(features.shape[1]), feature_dim=int(features.shape[2]), name='aim collated features')
    assert_shape(aim_delta_targets, (batch_size, 4), 'aim collated aim_delta_targets')
    assert_shape(binary_action_targets, (batch_size, 3), 'aim collated binary_action_targets')
    assert_shape(valid_aim_mask, (batch_size, 1), 'aim collated valid_aim_mask')
    metas = [item[4] for item in batch]
    return features, aim_delta_targets, binary_action_targets, valid_aim_mask, metas


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


def append_epoch_summary(log_path: Path, epoch_summary: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(epoch_summary, ensure_ascii=True) + '\n')


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


def compute_binary_action_stats(dataset) -> dict[str, object]:
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        source_dataset = dataset.dataset
        indices = [int(idx) for idx in dataset.indices]
    else:
        source_dataset = dataset
        indices = list(range(len(dataset)))
    positives = np.zeros(3, dtype=np.float64)
    valid_sum = 0.0
    for idx in indices:
        target = source_dataset.build_target(sample_metadata=source_dataset.get_sample_metadata(idx))
        positives += target.binary_actions
        valid_sum += float(target.valid_aim_mask[0])
    total = max(len(indices), 1)
    return {
        'positive_ratios': (positives / total).astype(np.float64).tolist(),
        'positive_counts': positives.astype(np.int64).tolist(),
        'valid_aim_rate': float(valid_sum / total),
        'sample_count': int(len(indices)),
    }


def compute_binary_pos_weight(dataset, mode: str) -> np.ndarray | None:
    if mode == 'none':
        return None
    stats = compute_binary_action_stats(dataset)
    ratios = np.asarray(stats['positive_ratios'], dtype=np.float64)
    pos_weight = np.ones(3, dtype=np.float32)
    for idx, ratio in enumerate(ratios):
        ratio = float(np.clip(ratio, 1e-4, 1.0 - 1e-4))
        pos_weight[idx] = float(np.clip((1.0 - ratio) / ratio, 1.0, 25.0))
    return pos_weight


def save_checkpoint(
    save_path: Path,
    model: 'torch.nn.Module',
    args: argparse.Namespace,
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    dataset_label: str,
    schema: FeatureSchema,
    demo_names: list[str],
    *,
    train_round_usage: list[dict[str, object]],
    val_round_usage: list[dict[str, object]],
    round_dataset_format: str | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'aim_attention',
        'input_dim': schema.feature_dim,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'feature_schema': schema.to_metadata(),
        'dataset_source': dataset_label,
        'dataset_dir': str(resolve_shared_dataset_root(args, PROJECT_ROOT)),
        'dataset_subdir': args.dataset_subdir,
        'round_dataset_format': round_dataset_format,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
        'aim_feature_mode': args.aim_feature_mode,
        'aim_head_mode': args.aim_head_mode,
        'train_metrics': {k: v for k, v in train_metrics.items() if k not in AIM_METRIC_DICT_KEYS},
        'val_metrics': {k: v for k, v in val_metrics.items() if k not in AIM_METRIC_DICT_KEYS},
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a supervised aim model from clean_play_ticks')
    add_common_training_data_args(parser, project_root=PROJECT_ROOT, legacy_dataset_dir=True)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--split-mode', choices=['demo', 'round', 'random'], default='demo')
    parser.add_argument('--alive-only', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-samples-per-demo', type=int, default=None)
    parser.add_argument('--max-cached-demos', type=int, default=2)
    parser.add_argument('--show-index-progress', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--allow-no-spotted-enemy', action='store_true')
    parser.add_argument('--aim-feature-mode', choices=[AIM_FEATURE_MODE_DEMO_PROJECTED, AIM_FEATURE_MODE_VISION_LIKE], default=AIM_FEATURE_MODE_DEMO_PROJECTED)
    parser.add_argument('--aim-head-mode', choices=[AIM_HEAD_MODE_LEGACY, AIM_HEAD_MODE_MULTI_HEAD], default=AIM_HEAD_MODE_LEGACY)
    parser.add_argument('--aim-weight', type=float, default=1.0)
    parser.add_argument('--binary-weight', type=float, default=1.0)
    parser.add_argument('--confidence-weight', type=float, default=0.25)
    parser.add_argument('--binary-pos-weight-mode', choices=['auto', 'none'], default='auto')
    parser.add_argument('--vision-dropout-prob', type=float, default=0.15)
    parser.add_argument('--vision-noise-std', type=float, default=0.03)
    parser.add_argument('--vision-confidence-jitter', type=float, default=0.1)
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--tensorboard-run-name', type=str, default=None)
    parser.add_argument('--disable-tensorboard', action='store_true')
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'aim_bc.pt')
    parser.add_argument('--resume-from', type=Path, default=None)
    parser.add_argument('--stream-by-round', action='store_true')
    parser.add_argument('--epochs-per-round', type=int, default=1)
    parser.add_argument('--shuffle-rounds', action='store_true')
    parser.add_argument('--max-rounds', type=int, default=None)
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    return resolve_shared_dataset_root(args, PROJECT_ROOT)


def build_dataset(args: argparse.Namespace) -> AimSequenceTorchDataset:
    base_dataset = MultiDemoSequenceDataset(
        dataset_dir=resolve_dataset_root(args),
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
    print(f'Sequence samples built: {len(base_dataset)}')
    return AimSequenceTorchDataset(
        base_dataset,
        seq_len=args.seq_len,
        require_spotted_enemy=not args.allow_no_spotted_enemy,
        feature_mode=args.aim_feature_mode,
        vision_dropout_prob=args.vision_dropout_prob,
        vision_noise_std=args.vision_noise_std,
        vision_confidence_jitter=args.vision_confidence_jitter,
    )


def build_single_round_dataset(args: argparse.Namespace, round_file: Path) -> AimSequenceTorchDataset:
    base_dataset = SingleRoundSequenceDataset(
        round_file=round_file,
        dataset_subdir=args.dataset_subdir,
        seq_len=args.seq_len,
        stride=args.stride,
        alive_only=args.alive_only,
    )
    return AimSequenceTorchDataset(
        base_dataset,
        seq_len=args.seq_len,
        require_spotted_enemy=not args.allow_no_spotted_enemy,
        feature_mode=args.aim_feature_mode,
        vision_dropout_prob=args.vision_dropout_prob,
        vision_noise_std=args.vision_noise_std,
        vision_confidence_jitter=args.vision_confidence_jitter,
    )


def discover_round_training_files(args: argparse.Namespace) -> list[Path]:
    if args.dataset_subdir != 'rounds-dataset':
        raise ValueError('--stream-by-round currently supports only --dataset-subdir rounds-dataset')
    return list_parquet_files(resolve_dataset_root(args), args.dataset_subdir, recursive=True)


def build_round_uid_from_file(round_file: Path) -> str:
    demo_dir = round_file.parent.parent.name
    stem = round_file.stem
    if not stem.startswith('round_'):
        raise ValueError(f'Cannot infer round uid from filename: {round_file}')
    return make_round_uid(demo_dir, int(stem[len("round_"):]), round_file.name)


def run_round_stream_training(args: argparse.Namespace, device: str, runtime_info: dict[str, object]) -> int:
    round_files = discover_round_training_files(args)
    if not round_files:
        print(f'No round parquet files found under {resolve_dataset_root(args) / args.dataset_subdir}.')
        return 1

    feature_extractor = AimFeatureExtractor(seq_len=args.seq_len, feature_mode=args.aim_feature_mode)
    feature_schema = feature_extractor.schema()
    model = AimAttentionModel(input_dim=feature_extractor.feature_dim(), head_mode=args.aim_head_mode).to(device)
    load_checkpoint_if_available(model, args.resume_from, device)
    trainer = AimTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        head_mode=args.aim_head_mode,
        aim_weight=args.aim_weight,
        binary_weight=args.binary_weight,
        confidence_weight=args.confidence_weight,
        log_interval=args.log_interval,
        binary_pos_weight=None,
    )
    train_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=True)
    if int(train_loader_kwargs.get('num_workers', 0)) > 0:
        print('Stream-by-round mode uses num_workers=0 to avoid multiprocessing startup overhead and worker failures.', flush=True)
        train_loader_kwargs['num_workers'] = 0
        train_loader_kwargs.pop('persistent_workers', None)
        train_loader_kwargs.pop('prefetch_factor', None)
    dataset_root = resolve_dataset_root(args)
    dataset_label = build_dataset_label(args, PROJECT_ROOT)
    run_id = resolve_run_id(args, 'aim_stream')
    ledger = TrainingRoundLedger.load(args.rounds_ledger_path)
    trained_round_uids = (
        ledger.read_trained_round_uids(
            module_name='aim',
            model_name=f'aim_attention_{args.aim_head_mode}',
            checkpoint_path=str(args.save_path),
            match_mode=args.ledger_match_mode,
        )
        if args.skip_trained_rounds else set()
    )

    writer = None
    if args.disable_tensorboard:
        print('TensorBoard: disabled')
    elif tensorboard_available():
        writer, run_dir = create_summary_writer(
            runs_dir=args.runs_dir,
            run_name=args.tensorboard_run_name,
            default_prefix='aim_stream',
            save_path=args.save_path,
            config={'args': vars(args), 'device': device, 'dataset_source': dataset_label, 'round_count': len(round_files)},
        )
        if run_dir is not None:
            print(f'TensorBoard run: {run_dir}')
    else:
        print('TensorBoard: unavailable (install tensorboard to enable event logging)')

    print('train_aim.py --stream-by-round')
    print(f'Device: {device}')
    print(f'Dataset source: {dataset_label}')
    print(f'Round files discovered: {len(round_files)}')
    print(f'DataLoader workers: train={train_loader_kwargs["num_workers"]}')
    print(f'CUDA tuning: matmul={runtime_info["matmul_precision"]} cudnn_benchmark={runtime_info["cudnn_benchmark"]} tf32={runtime_info["tf32"]}')
    print(f'Feature dim: {feature_extractor.feature_dim()}')
    print(f'Aim feature mode: {args.aim_feature_mode}')
    print(f'Aim head mode: {args.aim_head_mode}')
    print(f'Save path: {args.save_path}')

    rounds_completed = 0
    cycle_index = 0
    last_train_metrics: dict[str, object] = trainer._empty_metrics()
    try:
        while True:
            cycle_index += 1
            cycle_rounds = list(round_files)
            if args.shuffle_rounds:
                random.Random(args.seed + cycle_index).shuffle(cycle_rounds)
            processed_in_cycle = 0
            for round_path in cycle_rounds:
                if args.max_rounds is not None and rounds_completed >= int(args.max_rounds):
                    print(f'Reached --max-rounds={args.max_rounds}. Stopping.')
                    return 0
                round_uid = build_round_uid_from_file(round_path)
                if args.skip_trained_rounds and round_uid in trained_round_uids:
                    continue
                print(f'Loading round: {round_path}')
                dataset = build_single_round_dataset(args, round_path)
                dataset_len = len(dataset)
                if dataset_len == 0:
                    print(f'Skipping round with zero aim-relevant samples: {round_path}')
                    continue
                train_stats = compute_binary_action_stats(dataset)
                train_pos_weight_np = compute_binary_pos_weight(dataset, args.binary_pos_weight_mode) if dataset_len > 0 else None
                trainer.binary_pos_weight = torch.tensor(train_pos_weight_np, dtype=torch.float32).to(device) if train_pos_weight_np is not None else None
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_aim_batch, **train_loader_kwargs)
                print(
                    f'Round {round_uid}: samples={dataset_len} valid_aim_rate={train_stats["valid_aim_rate"]:.4f} '
                    f'epochs_per_round={args.epochs_per_round}',
                    flush=True,
                )
                for local_epoch in range(1, int(args.epochs_per_round) + 1):
                    global_epoch = rounds_completed * int(args.epochs_per_round) + local_epoch
                    last_train_metrics = trainer.train_epoch(train_loader, epoch=global_epoch, writer=writer)
                    print(
                        f'Round {round_uid} | epoch {local_epoch}/{args.epochs_per_round} | '
                        f'loss={last_train_metrics["loss"]:.4f} aim={last_train_metrics["aim_loss"]:.4f} '
                        f'binary={last_train_metrics["binary_loss"]:.4f} conf={last_train_metrics["confidence_loss"]:.4f}',
                        flush=True,
                    )
                    log_scalar_dict(writer, 'round_train', last_train_metrics, global_epoch, ignored_keys=AIM_METRIC_DICT_KEYS)
                    if writer is not None:
                        writer.flush()
                round_usage = collect_round_usage(dataset)
                ledger.append_run_rounds(
                    run_id=run_id,
                    module_name='aim',
                    model_name=f'aim_attention_{args.aim_head_mode}',
                    checkpoint_path=str(args.save_path),
                    dataset_dir=str(dataset_root),
                    dataset_subdir=args.dataset_subdir,
                    split_mode='round_stream',
                    split='train',
                    round_usage=round_usage,
                )
                trained_round_uids.add(round_uid)
                save_checkpoint(
                    args.save_path,
                    model,
                    args,
                    last_train_metrics,
                    trainer._empty_metrics(),
                    dataset_label,
                    feature_schema,
                    dataset.base_dataset.get_demo_names(),
                    train_round_usage=round_usage,
                    val_round_usage=[],
                    round_dataset_format='rounds',
                )
                rounds_completed += 1
                processed_in_cycle += 1
                print(f'Completed round {round_uid}. Total rounds trained: {rounds_completed}', flush=True)
                del train_loader
                del dataset
            if args.skip_trained_rounds and processed_in_cycle == 0:
                print('No remaining rounds to train after --skip-trained-rounds filtering.')
                break
    except KeyboardInterrupt:
        print('Training stopped by user.', flush=True)
    finally:
        close_summary_writer(writer)
    return 0


def main() -> int:
    if not torch_available():
        print('PyTorch is not available. Install torch to use train_aim.py')
        return 0

    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    runtime_info = configure_torch_runtime(device)
    if args.stream_by_round:
        return run_round_stream_training(args, device, runtime_info)

    try:
        print('Building dataset...')
        dataset = build_dataset(args)
    except FileNotFoundError as exc:
        print(exc)
        print(f'No parquet files found under {resolve_dataset_root(args) / args.dataset_subdir}. Run parser/cleaner first.')
        return 1

    dataset_len = len(dataset)
    if dataset_len == 0:
        print('Aim training dataset is empty. Try smaller seq_len/stride or another demo set.')
        return 1
    if args.skip_trained_rounds:
        dataset, skip_info = filter_dataset_by_trained_rounds(
            dataset,
            ledger_path=args.rounds_ledger_path,
            module_name='aim',
            model_name=f'aim_attention_{args.aim_head_mode}',
            checkpoint_path=str(args.save_path),
            match_mode=args.ledger_match_mode,
        )
        print(f'total rounds before skip: {skip_info["total_rounds_before_skip"]}')
        print(f'skipped rounds count: {skip_info["skipped_rounds_count"]}')
        print(f'remaining rounds count: {skip_info["remaining_rounds_count"]}')
        if len(dataset) == 0:
            print('Dataset is empty after --skip-trained-rounds filtering.')
            return 1

    train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
    train_round_usage = collect_round_usage(train_dataset)
    val_round_usage = collect_round_usage(val_dataset)
    run_id = resolve_run_id(args, 'aim')
    dataset_root = resolve_dataset_root(args)
    ledger = TrainingRoundLedger.load(args.rounds_ledger_path)
    ledger.append_run_rounds(
        run_id=run_id,
        module_name='aim',
        model_name=f'aim_attention_{args.aim_head_mode}',
        checkpoint_path=str(args.save_path),
        dataset_dir=str(dataset_root),
        dataset_subdir=args.dataset_subdir,
        split_mode=args.split_mode,
        split='train',
        round_usage=train_round_usage,
    )
    ledger.append_run_rounds(
        run_id=run_id,
        module_name='aim',
        model_name=f'aim_attention_{args.aim_head_mode}',
        checkpoint_path=str(args.save_path),
        dataset_dir=str(dataset_root),
        dataset_subdir=args.dataset_subdir,
        split_mode=args.split_mode,
        split='val',
        round_usage=val_round_usage,
    )
    train_expected_counts = collect_expected_demo_counts(train_dataset)
    val_expected_counts = collect_expected_demo_counts(val_dataset)
    train_stats = compute_binary_action_stats(train_dataset)
    train_pos_weight_np = compute_binary_pos_weight(train_dataset, args.binary_pos_weight_mode) if len(train_dataset) > 0 else None
    print(f'Aim relevance-filtered samples: {dataset_len}')
    print(f'Valid aim rate: {train_stats["valid_aim_rate"]:.4f}')
    print('Binary target positive ratios:')
    for name, ratio in zip(AIM_BINARY_ACTION_NAMES, train_stats['positive_ratios'], strict=True):
        print(f'  {name}: {float(ratio):.4f}')

    train_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=True)
    val_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_aim_batch, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_aim_batch, **val_loader_kwargs)

    feature_extractor = AimFeatureExtractor(seq_len=args.seq_len, feature_mode=args.aim_feature_mode)
    feature_schema = feature_extractor.schema()
    model = AimAttentionModel(input_dim=feature_extractor.feature_dim(), head_mode=args.aim_head_mode).to(device)
    load_checkpoint_if_available(model, args.resume_from, device)
    trainer = AimTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        head_mode=args.aim_head_mode,
        aim_weight=args.aim_weight,
        binary_weight=args.binary_weight,
        confidence_weight=args.confidence_weight,
        log_interval=args.log_interval,
        binary_pos_weight=torch.tensor(train_pos_weight_np, dtype=torch.float32) if train_pos_weight_np is not None else None,
    )
    demo_names = dataset.base_dataset.get_demo_names()
    dataset_label = build_dataset_label(args, PROJECT_ROOT)
    epoch_log_path = args.save_path.with_name(f'{args.save_path.stem}_epoch_metrics.jsonl')
    writer = None

    if args.disable_tensorboard:
        print('TensorBoard: disabled')
    elif tensorboard_available():
        writer, run_dir = create_summary_writer(
            runs_dir=args.runs_dir,
            run_name=args.tensorboard_run_name,
            default_prefix='aim',
            save_path=args.save_path,
            config={'args': vars(args), 'device': device, 'dataset_source': dataset_label, 'demo_names': demo_names},
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
    print(f'DataLoader workers: train={train_loader_kwargs["num_workers"]} val={val_loader_kwargs["num_workers"]}')
    print(f'CUDA tuning: matmul={runtime_info["matmul_precision"]} cudnn_benchmark={runtime_info["cudnn_benchmark"]} tf32={runtime_info["tf32"]}')
    print(f'Feature dim: {feature_extractor.feature_dim()}')
    print(f'Aim feature mode: {args.aim_feature_mode}')
    print(f'Aim head mode: {args.aim_head_mode}')
    print(f'Save path: {args.save_path}')
    print(f'Epoch log: {epoch_log_path}')

    best_val_loss = math.inf
    best_train_metrics: dict[str, object] | None = None
    best_val_metrics: dict[str, object] | None = None
    try:
        for epoch in range(1, args.epochs + 1):
            print(f'Starting epoch {epoch}/{args.epochs}...')
            train_metrics = trainer.train_epoch(train_loader, epoch=epoch, writer=writer)
            val_metrics = trainer.eval_epoch(val_loader, epoch=epoch, writer=writer) if len(val_dataset) > 0 else dict(train_metrics, seen_sample_ids=set(), per_demo_loss={}, per_demo_seen_counts={})
            print(
                f'Epoch {epoch}/{args.epochs} | '
                f'train_loss={train_metrics["loss"]:.4f} '
                f'(aim={train_metrics["aim_loss"]:.4f}, binary={train_metrics["binary_loss"]:.4f}, conf={train_metrics["confidence_loss"]:.4f}) | '
                f'val_loss={val_metrics["loss"]:.4f} '
                f'(aim={val_metrics["aim_loss"]:.4f}, binary={val_metrics["binary_loss"]:.4f}, conf={val_metrics["confidence_loss"]:.4f})'
            )
            print(
                f'  mouse_mae=({train_metrics["mouse_dx_mae"]:.4f}, {train_metrics["mouse_dy_mae"]:.4f}) '
                f'angular_err={train_metrics["mean_angular_error"]:.4f} '
                f'fire_f1={train_metrics["fire_f1"]["fire"]:.4f} '
                f'fire_rate={train_metrics["predicted_fire_rate"]:.4f}/{train_metrics["target_fire_rate"]:.4f}'
            )
            train_coverage = build_coverage_summary(train_metrics, train_expected_counts)
            val_coverage = build_coverage_summary(val_metrics, val_expected_counts)
            print_coverage_summary('train', train_coverage)
            if val_expected_counts:
                print_coverage_summary('val', val_coverage)

            append_epoch_summary(
                epoch_log_path,
                {
                    'epoch': epoch,
                    'train': {**{k: v for k, v in train_metrics.items() if k not in {'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'}}, 'coverage': train_coverage},
                    'val': {**{k: v for k, v in val_metrics.items() if k not in {'seen_sample_ids', 'per_demo_loss', 'per_demo_seen_counts'}}, 'coverage': val_coverage},
                },
            )
            log_scalar_dict(writer, 'train', train_metrics, epoch, ignored_keys=AIM_METRIC_DICT_KEYS)
            log_scalar_dict(writer, 'val', val_metrics, epoch, ignored_keys=AIM_METRIC_DICT_KEYS)
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
                    train_round_usage=train_round_usage,
                    val_round_usage=val_round_usage,
                    round_dataset_format=getattr(dataset.base_dataset, 'dataset_format', None),
                )
                print(f'  saved checkpoint -> {args.save_path}')
    finally:
        close_summary_writer(writer)

    print('Training finished.')
    print(f'Best val loss: {best_val_loss:.4f}')
    if best_train_metrics is None or best_val_metrics is None:
        best_train_metrics = trainer._empty_metrics()
        best_val_metrics = trainer._empty_metrics()
    report = build_base_training_report(
        module_name='aim',
        model_name=f'aim_attention_{args.aim_head_mode}',
        dataset_path=dataset_label,
        split_mode=args.split_mode,
        seq_len=args.seq_len,
        feature_dim=feature_extractor.feature_dim(),
        target_shape='aim_delta:[batch,4], binary_actions:[batch,3], valid_aim_mask:[batch,1]',
        checkpoint_path=str(args.save_path),
        config=vars(args),
        train_metrics=best_train_metrics,
        val_metrics=best_val_metrics,
    )
    report_paths = write_training_report(report)
    print(f'Reports: {report_paths["json"]} | {report_paths["csv"]} | {report_paths["markdown"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
