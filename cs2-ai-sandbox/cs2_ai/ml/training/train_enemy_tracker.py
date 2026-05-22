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

from cs2_ai.config import MAX_ENEMIES
from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from cs2_ai.dataset.parquet_loader import list_parquet_files, load_parquet, parquet_demo_name
from cs2_ai.dataset.round_identity import make_round_uid
from cs2_ai.features.enemy_tracker_features import (
    TRACKER_FEATURE_NAMES,
    EnemyTrackerFeatureExtractor,
    build_enemy_confidence_target,
    build_enemy_position_target,
    build_enemy_roster,
)
from cs2_ai.features.feature_contract import FeatureSchema
from cs2_ai.ml.models.enemy_tracker_lstm import (
    ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK,
    ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK,
    EnemyTrackerLSTM,
)
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


LAST_SEEN_BUCKET_VISIBLE = 0
LAST_SEEN_BUCKET_AGE_1_2 = 1
LAST_SEEN_BUCKET_AGE_3_5 = 2
LAST_SEEN_BUCKET_AGE_6_PLUS = 3
LAST_SEEN_BUCKET_UNAVAILABLE = 4
LAST_SEEN_BUCKET_LABELS = {
    LAST_SEEN_BUCKET_VISIBLE: 'visible',
    LAST_SEEN_BUCKET_AGE_1_2: 'age_1_2',
    LAST_SEEN_BUCKET_AGE_3_5: 'age_3_5',
    LAST_SEEN_BUCKET_AGE_6_PLUS: 'age_6_plus',
    LAST_SEEN_BUCKET_UNAVAILABLE: 'unavailable',
}
TRACKER_METRIC_DICT_KEYS = {
    'seen_sample_ids',
    'per_demo_loss',
    'per_demo_seen_counts',
    'confidence_precision',
    'confidence_recall',
    'confidence_f1',
    'last_seen_bucket_distance_error',
}

ROUND_COLUMN_CANDIDATES = ['round_number', 'total_rounds_played']


@dataclass(slots=True)
class TrackerTrainingBatch:
    features: 'torch.Tensor'
    target_positions: 'torch.Tensor'
    target_confidences: 'torch.Tensor'
    age_bucket_ids: 'torch.Tensor'
    sample_ids: list[str]
    demo_names: list[str]


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
            return int(stem[len('round_'):])
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

    def build_truth_state_for_sample_tick(self, sample_metadata: dict[str, object], tick: int):
        return self.build_bundle_for_sample_tick(sample_metadata, tick).truth_state

    def __getitem__(self, idx: int) -> SequenceSample:
        sample_index = self.get_sample_metadata(idx)
        round_number = int(sample_index['round_number'])
        perspective_steamid = int(sample_index['perspective_steamid'])
        tick_indices = list(sample_index['tick_indices'])
        target_tick = int(sample_index['target_tick'])
        states = [self.build_bundle_for_sample_tick(sample_index, int(tick)).observed_state for tick in tick_indices]
        target_state = self.build_bundle_for_sample_tick(sample_index, target_tick).observed_state
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


def _enemy_slot_feature_indices() -> dict[int, tuple[int, int, int]]:
    mapping: dict[int, tuple[int, int, int]] = {}
    for slot in range(MAX_ENEMIES):
        mapping[slot] = (
            TRACKER_FEATURE_NAMES.index(f'enemy_{slot}_visible_mask'),
            TRACKER_FEATURE_NAMES.index(f'enemy_{slot}_last_seen_mask'),
            TRACKER_FEATURE_NAMES.index(f'enemy_{slot}_unavailable_mask'),
        )
    return mapping


ENEMY_SLOT_FEATURE_INDICES = _enemy_slot_feature_indices()


def infer_last_seen_bucket_ids(features: np.ndarray, output_mode: str) -> np.ndarray:
    seq_len = int(features.shape[0])
    bucket_matrix = np.full((seq_len, MAX_ENEMIES), LAST_SEEN_BUCKET_UNAVAILABLE, dtype=np.int64)
    for slot in range(MAX_ENEMIES):
        visible_idx, last_seen_idx, unavailable_idx = ENEMY_SLOT_FEATURE_INDICES[slot]
        for step_idx in range(seq_len):
            visible_value = float(features[step_idx, visible_idx])
            last_seen_value = float(features[step_idx, last_seen_idx])
            unavailable_value = float(features[step_idx, unavailable_idx])
            if visible_value >= 0.5:
                bucket_matrix[step_idx, slot] = LAST_SEEN_BUCKET_VISIBLE
                continue
            if unavailable_value >= 0.5:
                bucket_matrix[step_idx, slot] = LAST_SEEN_BUCKET_UNAVAILABLE
                continue
            if last_seen_value < 0.5:
                bucket_matrix[step_idx, slot] = LAST_SEEN_BUCKET_UNAVAILABLE
                continue
            last_visible_idx = None
            for lookback_idx in range(step_idx - 1, -1, -1):
                if float(features[lookback_idx, visible_idx]) >= 0.5:
                    last_visible_idx = lookback_idx
                    break
            age_steps = (step_idx - last_visible_idx) if last_visible_idx is not None else (step_idx + 1)
            if age_steps <= 2:
                bucket_matrix[step_idx, slot] = LAST_SEEN_BUCKET_AGE_1_2
            elif age_steps <= 5:
                bucket_matrix[step_idx, slot] = LAST_SEEN_BUCKET_AGE_3_5
            else:
                bucket_matrix[step_idx, slot] = LAST_SEEN_BUCKET_AGE_6_PLUS
    if output_mode == ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK:
        return bucket_matrix[-1]
    return bucket_matrix


class EnemyTrackerSequenceTorchDataset(Dataset):
    def __init__(self, base_dataset, seq_len: int, output_mode: str = ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK):
        if output_mode not in {ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK, ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK}:
            raise ValueError(f'Unsupported enemy tracker output mode: {output_mode}')
        self.base_dataset = base_dataset
        self.seq_len = int(seq_len)
        self.output_mode = output_mode
        self.feature_extractor = EnemyTrackerFeatureExtractor(seq_len=seq_len)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def get_sample_metadata(self, idx: int) -> dict[str, object]:
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        return ds.get_sample_metadata(real_idx)

    @property
    def output_len(self) -> int:
        return self.seq_len if self.output_mode == ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK else 1

    def _build_fast_target(self, ds, round_number, tick, perspective_steamid, roster_steamids):
        import pandas as pd
        try:
            tick_rows = ds.round_tick_rows[round_number][tick]
        except Exception:
            return None, None
            
        if 'steamid' not in tick_rows.columns or 'team_num' not in tick_rows.columns or 'X' not in tick_rows.columns:
            return None, None
            
        steamids = pd.to_numeric(tick_rows['steamid'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
        team_values = pd.to_numeric(tick_rows['team_num'], errors='coerce').to_numpy(dtype=np.float64)
        
        self_mask = steamids == perspective_steamid
        if not np.any(self_mask):
            return None, None
            
        self_team = team_values[self_mask][0]
        if np.isnan(self_team):
            return None, None
            
        if 'is_alive' in tick_rows.columns:
            is_alive = tick_rows['is_alive'].fillna(False).astype(bool).to_numpy(dtype=bool)
        else:
            is_alive = np.ones(len(tick_rows), dtype=bool)
            
        x_vals = pd.to_numeric(tick_rows['X'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        y_vals = pd.to_numeric(tick_rows['Y'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        z_vals = pd.to_numeric(tick_rows['Z'], errors='coerce').fillna(0).to_numpy(dtype=np.float32)

        target_pos = np.zeros((MAX_ENEMIES, 3), dtype=np.float32)
        target_conf = np.zeros(MAX_ENEMIES, dtype=np.float32)
        
        roster_map = {steamid: i for i, steamid in enumerate(roster_steamids)}
        
        for i in range(len(steamids)):
            steamid = steamids[i]
            if steamid == -1: continue
            
            team = team_values[i]
            if np.isnan(team) or team == self_team:
                continue
                
            enemy_idx = roster_map.get(steamid)
            if enemy_idx is not None:
                target_pos[enemy_idx, 0] = x_vals[i]
                target_pos[enemy_idx, 1] = y_vals[i]
                target_pos[enemy_idx, 2] = z_vals[i]
                if is_alive[i]:
                    target_conf[enemy_idx] = 1.0
                    
        return target_pos, target_conf

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        import time
        t_start = time.time()
        sequence_sample = self.base_dataset[idx]
        t_base = time.time()
        features = self.feature_extractor.extract(sequence_sample.sequence).astype(np.float32)
        t_feat = time.time()
        
        assert_shape(features, (self.seq_len, self.feature_extractor.feature_dim()), 'enemy tracker sample features')
        ds, real_idx = get_base_dataset_and_index(self.base_dataset, idx)
        sample_metadata = ds.get_sample_metadata(real_idx)
        tick_indices = list(sample_metadata['tick_indices'])
        target_tick = int(sample_metadata['target_tick'])
        perspective_steamid = int(sample_metadata['perspective_steamid'])
        round_number = int(sample_metadata['round_number'])
        roster_steamids = build_enemy_roster(sequence_sample.sequence)

        can_fast_path = type(ds).__name__ == 'SingleRoundSequenceDataset' and hasattr(ds, 'round_tick_rows')

        if self.output_mode == ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK:
            tp, tc = None, None
            if can_fast_path:
                tp, tc = self._build_fast_target(ds, round_number, target_tick, perspective_steamid, roster_steamids)
            if tp is None:
                target_state = ds.build_truth_state_for_sample_tick(sample_metadata, target_tick)
                tp = build_enemy_position_target(target_state, roster_steamids)
                tc = build_enemy_confidence_target(target_state, roster_steamids)
            target_positions = tp
            target_confidences = tc
            age_bucket_ids = infer_last_seen_bucket_ids(features, self.output_mode)
            assert_shape(target_positions, (MAX_ENEMIES, 3), 'enemy tracker sample target positions')
            assert_shape(target_confidences, (MAX_ENEMIES,), 'enemy tracker sample target confidences')
            assert_shape(age_bucket_ids, (MAX_ENEMIES,), 'enemy tracker sample age buckets')
        else:
            target_ticks = tick_indices[1:] + [target_tick]
            target_positions = np.zeros((len(target_ticks), MAX_ENEMIES, 3), dtype=np.float32)
            target_confidences = np.zeros((len(target_ticks), MAX_ENEMIES), dtype=np.float32)
            for t_idx, tick in enumerate(target_ticks):
                tp, tc = None, None
                if can_fast_path:
                    tp, tc = self._build_fast_target(ds, round_number, tick, perspective_steamid, roster_steamids)
                if tp is None:
                    target_state = ds.build_truth_state_for_sample_tick(sample_metadata, tick)
                    tp = build_enemy_position_target(target_state, roster_steamids)
                    tc = build_enemy_confidence_target(target_state, roster_steamids)
                target_positions[t_idx] = tp
                target_confidences[t_idx] = tc
            age_bucket_ids = infer_last_seen_bucket_ids(features, self.output_mode)
            assert_shape(target_positions, (self.seq_len, MAX_ENEMIES, 3), 'enemy tracker sample target positions')
            assert_shape(target_confidences, (self.seq_len, MAX_ENEMIES), 'enemy tracker sample target confidences')
            assert_shape(age_bucket_ids, (self.seq_len, MAX_ENEMIES), 'enemy tracker sample age buckets')

        t_targ = time.time()
        if not hasattr(self, '_profile_count'):
            self._profile_count = 0
            self._profile_stats = {'base': 0.0, 'feat': 0.0, 'targ': 0.0}
            
        if self._profile_count < 50:
            self._profile_stats['base'] += t_base - t_start
            self._profile_stats['feat'] += t_feat - t_base
            self._profile_stats['targ'] += t_targ - t_feat
            self._profile_count += 1
            if self._profile_count == 50:
                print(f"EnemyTracker DataLoader Profiling (avg over 50): Base={self._profile_stats['base']/50:.4f}s, Feat={self._profile_stats['feat']/50:.4f}s, Targ={self._profile_stats['targ']/50:.4f}s", flush=True)

        meta = {
            'sample_id': str(sample_metadata['sample_id']),
            'demo_name': str(sample_metadata['demo_name']),
            'roster_steamids': [int(steamid) for steamid in roster_steamids],
        }
        return features, target_positions.astype(np.float32), target_confidences.astype(np.float32), age_bucket_ids.astype(np.int64), meta


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
        total_distance_error = 0.0
        total_distance_count = 0.0
        total_top1_error = 0.0
        total_top1_count = 0.0
        conf_tp = 0.0
        conf_fp = 0.0
        conf_fn = 0.0
        bucket_error_sum = {label: 0.0 for label in LAST_SEEN_BUCKET_LABELS.values()}
        bucket_error_count = {label: 0.0 for label in LAST_SEEN_BUCKET_LABELS.values()}

        print(f'{phase_name} epoch {epoch} | Preparing first batch...')

        for batch_idx, batch in enumerate(loader):
            if not first_batch_loaded:
                print(f'{phase_name} epoch {epoch} | First batch loaded.')
                first_batch_loaded = True
            batch = self._to_training_batch(batch)
            assert_temporal_features(
                batch.features,
                seq_len=batch.features.shape[1],
                feature_dim=batch.features.shape[2],
                name='enemy tracker batch features',
            )
            pred_positions, pred_confidences = self.model(batch.features)
            pred_positions, pred_confidences = self._normalize_outputs(
                pred_positions,
                pred_confidences,
                batch.target_positions,
                batch.target_confidences,
            )

            target_mask = (batch.target_confidences > 0.5).to(dtype=batch.target_positions.dtype)
            pos_error_raw = F.smooth_l1_loss(pred_positions, batch.target_positions, reduction='none').mean(dim=-1)
            pos_loss_per_sample = self._masked_mean(pos_error_raw, target_mask)
            conf_loss_raw = F.binary_cross_entropy_with_logits(pred_confidences, batch.target_confidences, reduction='none')
            conf_loss_per_sample = conf_loss_raw.reshape(conf_loss_raw.shape[0], -1).mean(dim=1)
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

            conf_probs = torch.sigmoid(pred_confidences)
            pred_binary = (conf_probs > 0.5).to(dtype=batch.target_confidences.dtype)
            target_binary = (batch.target_confidences > 0.5).to(dtype=batch.target_confidences.dtype)
            conf_tp += float((pred_binary * target_binary).sum().item())
            conf_fp += float((pred_binary * (1.0 - target_binary)).sum().item())
            conf_fn += float(((1.0 - pred_binary) * target_binary).sum().item())

            distances = torch.sqrt(torch.clamp(((pred_positions - batch.target_positions) ** 2).sum(dim=-1), min=0.0))
            total_distance_error += float((distances * target_mask).sum().item())
            total_distance_count += float(target_mask.sum().item())

            top1_error_sum, top1_count = self._compute_top1_error(distances, conf_probs, target_mask)
            total_top1_error += top1_error_sum
            total_top1_count += top1_count

            batch_bucket_sum, batch_bucket_count = self._accumulate_bucket_errors(distances, target_mask, batch.age_bucket_ids)
            for bucket_name in bucket_error_sum:
                bucket_error_sum[bucket_name] += batch_bucket_sum[bucket_name]
                bucket_error_count[bucket_name] += batch_bucket_count[bucket_name]

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
                print(
                    f'{phase_name} epoch {epoch} | Batch {batch_idx + 1}/{total_batches} | '
                    f'Loss: {loss.item():.4f} | Seen: {len(seen_sample_ids)}'
                )

        if total_samples == 0:
            return {
                'loss': 0.0,
                'pos_loss': 0.0,
                'conf_loss': 0.0,
                'mean_distance_error': 0.0,
                'top1_enemy_position_error': 0.0,
                'confidence_precision': {},
                'confidence_recall': {},
                'confidence_f1': {},
                'last_seen_bucket_distance_error': {},
                'seen_sample_ids': set(),
                'per_demo_loss': {},
                'per_demo_seen_counts': {},
            }

        precision = float(conf_tp / max(conf_tp + conf_fp, 1.0))
        recall = float(conf_tp / max(conf_tp + conf_fn, 1.0))
        f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-8))
        bucket_means = {
            name: float(bucket_error_sum[name] / bucket_error_count[name]) if bucket_error_count[name] > 0 else 0.0
            for name in bucket_error_sum
        }
        return {
            'loss': total_loss / total_samples,
            'pos_loss': total_pos_loss / total_samples,
            'conf_loss': total_conf_loss / total_samples,
            'mean_distance_error': float(total_distance_error / max(total_distance_count, 1.0)),
            'top1_enemy_position_error': float(total_top1_error / max(total_top1_count, 1.0)),
            'confidence_precision': {'known_enemy': precision},
            'confidence_recall': {'known_enemy': recall},
            'confidence_f1': {'known_enemy': f1},
            'last_seen_bucket_distance_error': bucket_means,
            'seen_sample_ids': seen_sample_ids,
            'per_demo_loss': {demo: per_demo_sum[demo] / per_demo_count[demo] for demo in sorted(per_demo_sum)},
            'per_demo_seen_counts': {demo: len(ids) for demo, ids in sorted(seen_demo_sample_ids.items())},
        }

    def _normalize_outputs(
        self,
        pred_positions: 'torch.Tensor',
        pred_confidences: 'torch.Tensor',
        target_positions: 'torch.Tensor',
        target_confidences: 'torch.Tensor',
    ) -> tuple['torch.Tensor', 'torch.Tensor']:
        if pred_positions.ndim == 4:
            assert_shape(pred_positions, tuple(target_positions.shape), 'enemy tracker predicted positions')
            assert_shape(pred_confidences, tuple(target_confidences.shape), 'enemy tracker predicted confidences')
            return pred_positions, pred_confidences
        if pred_positions.ndim == 3:
            assert_shape(pred_positions, tuple(target_positions.shape), 'enemy tracker predicted positions')
            assert_shape(pred_confidences, tuple(target_confidences.shape), 'enemy tracker predicted confidences')
            return pred_positions, pred_confidences
        raise ValueError(
            f'Enemy tracker outputs must be rank-4/3 for positions and rank-3/2 for confidences, got '
            f'positions={tuple(pred_positions.shape)} confidences={tuple(pred_confidences.shape)}.'
        )

    def _masked_mean(self, values: 'torch.Tensor', mask: 'torch.Tensor') -> 'torch.Tensor':
        flat_values = values.reshape(values.shape[0], -1)
        flat_mask = mask.reshape(mask.shape[0], -1)
        denom = torch.clamp(flat_mask.sum(dim=1), min=1.0)
        return (flat_values * flat_mask).sum(dim=1) / denom

    def _compute_top1_error(
        self,
        distances: 'torch.Tensor',
        conf_probs: 'torch.Tensor',
        target_mask: 'torch.Tensor',
    ) -> tuple[float, float]:
        if distances.ndim == 3:
            masked_probs = conf_probs.masked_fill(target_mask <= 0.0, -1.0)
            best_idx = masked_probs.argmax(dim=-1)
            has_any = (target_mask.sum(dim=-1) > 0.0).to(dtype=distances.dtype)
            gathered = distances.gather(dim=-1, index=best_idx.unsqueeze(-1)).squeeze(-1)
            return float((gathered * has_any).sum().item()), float(has_any.sum().item())
        masked_probs = conf_probs.masked_fill(target_mask <= 0.0, -1.0)
        best_idx = masked_probs.argmax(dim=-1)
        has_any = (target_mask.sum(dim=-1) > 0.0).to(dtype=distances.dtype)
        gathered = distances.gather(dim=-1, index=best_idx.unsqueeze(-1)).squeeze(-1)
        return float((gathered * has_any).sum().item()), float(has_any.sum().item())

    def _accumulate_bucket_errors(
        self,
        distances: 'torch.Tensor',
        target_mask: 'torch.Tensor',
        age_bucket_ids: 'torch.Tensor',
    ) -> tuple[dict[str, float], dict[str, float]]:
        bucket_sum = {label: 0.0 for label in LAST_SEEN_BUCKET_LABELS.values()}
        bucket_count = {label: 0.0 for label in LAST_SEEN_BUCKET_LABELS.values()}
        for bucket_id, bucket_name in LAST_SEEN_BUCKET_LABELS.items():
            bucket_mask = (age_bucket_ids == bucket_id).to(dtype=target_mask.dtype) * target_mask
            bucket_sum[bucket_name] += float((distances * bucket_mask).sum().item())
            bucket_count[bucket_name] += float(bucket_mask.sum().item())
        return bucket_sum, bucket_count

    def _to_training_batch(
        self,
        batch: tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, object]]],
    ) -> TrackerTrainingBatch:
        features, target_pos, target_conf, age_bucket_ids, metas = batch
        return TrackerTrainingBatch(
            features=features.to(self.device, non_blocking=True),
            target_positions=target_pos.to(self.device, non_blocking=True),
            target_confidences=target_conf.to(self.device, non_blocking=True),
            age_bucket_ids=age_bucket_ids.to(self.device, non_blocking=True),
            sample_ids=[str(meta['sample_id']) for meta in metas],
            demo_names=[str(meta['demo_name']) for meta in metas],
        )


def collate_tracker_batch(
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]
) -> tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor', list[dict[str, object]]]:
    features = torch.from_numpy(np.stack([item[0] for item in batch]).astype(np.float32, copy=False))
    target_pos = torch.from_numpy(np.stack([item[1] for item in batch]).astype(np.float32, copy=False))
    target_conf = torch.from_numpy(np.stack([item[2] for item in batch]).astype(np.float32, copy=False))
    age_bucket_ids = torch.from_numpy(np.stack([item[3] for item in batch]).astype(np.int64, copy=False))
    metas = [item[4] for item in batch]
    return features, target_pos, target_conf, age_bucket_ids, metas


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
    *,
    train_round_usage: list[dict[str, object]],
    val_round_usage: list[dict[str, object]],
    round_dataset_format: str | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': 'enemy_tracker_lstm',
        'input_dim': schema.feature_dim,
        'seq_len': args.seq_len,
        'stride': args.stride,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'output_enemies': MAX_ENEMIES,
        'output_mode': args.output_mode,
        'feature_schema': schema.to_metadata(),
        'dataset_source': dataset_label,
        'dataset_dir': str(resolve_shared_dataset_root(args, PROJECT_ROOT)),
        'dataset_subdir': args.dataset_subdir,
        'round_dataset_format': round_dataset_format,
        'demo_names': demo_names,
        'demo_count': len(demo_names),
        'split_mode': args.split_mode,
        'train_metrics': {k: v for k, v in train_metrics.items() if k not in TRACKER_METRIC_DICT_KEYS},
        'val_metrics': {k: v for k, v in val_metrics.items() if k not in TRACKER_METRIC_DICT_KEYS},
        'train_round_count': len(train_round_usage),
        'val_round_count': len(val_round_usage),
        'train_round_uids': [item['round_uid'] for item in train_round_usage],
        'val_round_uids': [item['round_uid'] for item in val_round_usage],
        'rounds_ledger_path': str(args.rounds_ledger_path),
    }
    torch.save(checkpoint, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train supervised enemy tracker from rounds-dataset or clean_play_ticks')
    add_common_training_data_args(parser, project_root=PROJECT_ROOT, legacy_dataset_dir=True)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--output-mode', choices=[ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK, ENEMY_TRACKER_OUTPUT_MODE_TARGET_TICK], default=ENEMY_TRACKER_OUTPUT_MODE_EACH_TICK)
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
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--tensorboard-run-name', type=str, default=None)
    parser.add_argument('--disable-tensorboard', action='store_true')
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'enemy_tracker_bc.pt')
    parser.add_argument('--stream-by-round', action='store_true')
    parser.add_argument('--epochs-per-round', type=int, default=1)
    parser.add_argument('--shuffle-rounds', action='store_true')
    parser.add_argument('--max-rounds', type=int, default=None)
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    return resolve_shared_dataset_root(args, PROJECT_ROOT)


def build_dataset(args: argparse.Namespace) -> EnemyTrackerSequenceTorchDataset:
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
    return EnemyTrackerSequenceTorchDataset(base_dataset, seq_len=args.seq_len, output_mode=args.output_mode)


def build_single_round_dataset(args: argparse.Namespace, round_file: Path) -> EnemyTrackerSequenceTorchDataset:
    base_dataset = SingleRoundSequenceDataset(
        round_file=round_file,
        dataset_subdir=args.dataset_subdir,
        seq_len=args.seq_len,
        stride=args.stride,
        alive_only=args.alive_only,
    )
    return EnemyTrackerSequenceTorchDataset(base_dataset, seq_len=args.seq_len, output_mode=args.output_mode)


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

    feature_extractor = EnemyTrackerFeatureExtractor(seq_len=args.seq_len)
    feature_schema = feature_extractor.schema()
    model = EnemyTrackerLSTM(
        input_dim=feature_extractor.feature_dim(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_enemies=MAX_ENEMIES,
        dropout=args.dropout,
        output_mode=args.output_mode,
    ).to(device)
    trainer = EnemyTrackerTrainer(model=model, device=device, learning_rate=args.lr, log_interval=args.log_interval)
    train_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=True)
    if int(train_loader_kwargs.get('num_workers', 0)) > 0:
        print('Stream-by-round mode uses num_workers=0 to avoid multiprocessing startup overhead and worker failures.', flush=True)
        train_loader_kwargs['num_workers'] = 0
        train_loader_kwargs.pop('persistent_workers', None)
        train_loader_kwargs.pop('prefetch_factor', None)
    dataset_root = resolve_dataset_root(args)
    dataset_label = build_dataset_label(args, PROJECT_ROOT)
    run_id = resolve_run_id(args, 'enemy_tracker_stream')
    ledger = TrainingRoundLedger.load(args.rounds_ledger_path)
    trained_round_uids = (
        ledger.read_trained_round_uids(
            module_name='enemy_tracker',
            model_name=f'enemy_tracker_lstm_{args.output_mode}',
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
            default_prefix='enemy_tracker_stream',
            save_path=args.save_path,
            config={'args': vars(args), 'device': device, 'dataset_source': dataset_label, 'round_count': len(round_files)},
        )
        if run_dir is not None:
            print(f'TensorBoard run: {run_dir}')
    else:
        print('TensorBoard: unavailable (install tensorboard to enable event logging)')

    print('train_enemy_tracker.py --stream-by-round')
    print(f'Device: {device}')
    print(f'Dataset source: {dataset_label}')
    print(f'Round files discovered: {len(round_files)}')
    print(f'DataLoader workers: train={train_loader_kwargs["num_workers"]}')
    print(f'CUDA tuning: matmul={runtime_info["matmul_precision"]} cudnn_benchmark={runtime_info["cudnn_benchmark"]} tf32={runtime_info["tf32"]}')
    print(f'Feature dim: {feature_extractor.feature_dim()}')
    print(f'Output mode: {args.output_mode}')
    print(f'Save path: {args.save_path}')

    rounds_completed = 0
    cycle_index = 0
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
                    print(f'Skipping round with zero tracker samples: {round_path}')
                    continue
                train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tracker_batch, **train_loader_kwargs)
                print(
                    f'Round {round_uid}: samples={dataset_len} epochs_per_round={args.epochs_per_round}',
                    flush=True,
                )
                last_train_metrics: dict[str, object] | None = None
                for local_epoch in range(1, int(args.epochs_per_round) + 1):
                    global_epoch = rounds_completed * int(args.epochs_per_round) + local_epoch
                    last_train_metrics = trainer.train_epoch(train_loader, epoch=global_epoch, writer=writer)
                    print(
                        f'Round {round_uid} | epoch {local_epoch}/{args.epochs_per_round} | '
                        f'loss={last_train_metrics["loss"]:.4f} pos={last_train_metrics["pos_loss"]:.4f} '
                        f'conf={last_train_metrics["conf_loss"]:.4f}',
                        flush=True,
                    )
                if last_train_metrics is None:
                    continue
                log_scalar_dict(writer, 'round_train', last_train_metrics, rounds_completed + 1, ignored_keys=TRACKER_METRIC_DICT_KEYS)
                if writer is not None:
                    writer.flush()
                round_usage = collect_round_usage(dataset)
                checkpoint_val_metrics = {
                    'loss': last_train_metrics['loss'],
                    'pos_loss': last_train_metrics['pos_loss'],
                    'conf_loss': last_train_metrics['conf_loss'],
                    'mean_distance_error': last_train_metrics['mean_distance_error'],
                    'top1_enemy_position_error': last_train_metrics['top1_enemy_position_error'],
                    'confidence_precision': {},
                    'confidence_recall': {},
                    'confidence_f1': {},
                    'last_seen_bucket_distance_error': {},
                    'seen_sample_ids': set(),
                    'per_demo_loss': {},
                    'per_demo_seen_counts': {},
                }
                ledger.append_run_rounds(
                    run_id=run_id,
                    module_name='enemy_tracker',
                    model_name=f'enemy_tracker_lstm_{args.output_mode}',
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
                    checkpoint_val_metrics,
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
        print('PyTorch is not available. Install torch to use train_enemy_tracker.py')
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
        print('Tracker training dataset is empty.')
        return 1
    if args.skip_trained_rounds:
        dataset, skip_info = filter_dataset_by_trained_rounds(
            dataset,
            ledger_path=args.rounds_ledger_path,
            module_name='enemy_tracker',
            model_name=f'enemy_tracker_lstm_{args.output_mode}',
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
    train_dataset, val_dataset = split_dataset_by_group(dataset, args.val_split, args.seed, mode=args.split_mode)
    train_round_usage = collect_round_usage(train_dataset)
    val_round_usage = collect_round_usage(val_dataset)
    ledger = TrainingRoundLedger.load(args.rounds_ledger_path)
    run_id = resolve_run_id(args, 'enemy_tracker')
    dataset_root = resolve_dataset_root(args)
    ledger.append_run_rounds(
        run_id=run_id,
        module_name='enemy_tracker',
        model_name=f'enemy_tracker_lstm_{args.output_mode}',
        checkpoint_path=str(args.save_path),
        dataset_dir=str(dataset_root),
        dataset_subdir=args.dataset_subdir,
        split_mode=args.split_mode,
        split='train',
        round_usage=train_round_usage,
    )
    ledger.append_run_rounds(
        run_id=run_id,
        module_name='enemy_tracker',
        model_name=f'enemy_tracker_lstm_{args.output_mode}',
        checkpoint_path=str(args.save_path),
        dataset_dir=str(dataset_root),
        dataset_subdir=args.dataset_subdir,
        split_mode=args.split_mode,
        split='val',
        round_usage=val_round_usage,
    )
    train_expected_counts = collect_expected_demo_counts(train_dataset)
    val_expected_counts = collect_expected_demo_counts(val_dataset)
    print('Preparing dataloaders...')
    train_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=True)
    val_loader_kwargs = build_dataloader_kwargs(device, args.num_workers, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tracker_batch, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tracker_batch, **val_loader_kwargs)

    print('Initializing model and trainer...')
    feature_extractor = EnemyTrackerFeatureExtractor(seq_len=args.seq_len)
    feature_schema = feature_extractor.schema()
    model = EnemyTrackerLSTM(
        input_dim=feature_extractor.feature_dim(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_enemies=MAX_ENEMIES,
        dropout=args.dropout,
        output_mode=args.output_mode,
    ).to(device)
    trainer = EnemyTrackerTrainer(model=model, device=device, learning_rate=args.lr, log_interval=args.log_interval)
    demo_names = dataset.base_dataset.get_demo_names()
    dataset_label = build_dataset_label(args, PROJECT_ROOT)
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
    print(f'Output mode: {args.output_mode}')
    print(f'DataLoader workers: train={train_loader_kwargs["num_workers"]} val={val_loader_kwargs["num_workers"]}')
    print(f'CUDA tuning: matmul={runtime_info["matmul_precision"]} cudnn_benchmark={runtime_info["cudnn_benchmark"]} tf32={runtime_info["tf32"]}')
    print(f'Feature dim: {feature_extractor.feature_dim()}')
    print(f'Save path: {args.save_path}')
    print(f'Epoch log: {epoch_log_path}')

    best_val_loss = math.inf
    best_train_metrics: dict[str, object] | None = None
    best_val_metrics: dict[str, object] | None = None
    try:
        for epoch in range(1, args.epochs + 1):
            print(f'Starting epoch {epoch}/{args.epochs}...')
            train_metrics = trainer.train_epoch(train_loader, epoch=epoch, writer=writer)
            val_metrics = trainer.eval_epoch(val_loader, epoch=epoch, writer=writer) if len(val_dataset) > 0 else {
                'loss': train_metrics['loss'],
                'pos_loss': train_metrics['pos_loss'],
                'conf_loss': train_metrics['conf_loss'],
                'mean_distance_error': train_metrics['mean_distance_error'],
                'top1_enemy_position_error': train_metrics['top1_enemy_position_error'],
                'confidence_precision': dict(train_metrics['confidence_precision']),
                'confidence_recall': dict(train_metrics['confidence_recall']),
                'confidence_f1': dict(train_metrics['confidence_f1']),
                'last_seen_bucket_distance_error': dict(train_metrics['last_seen_bucket_distance_error']),
                'seen_sample_ids': set(),
                'per_demo_loss': {},
                'per_demo_seen_counts': {},
            }
            print(
                f'Epoch {epoch}/{args.epochs} | '
                f'train_loss={train_metrics["loss"]:.4f} '
                f'(pos={train_metrics["pos_loss"]:.4f}, conf={train_metrics["conf_loss"]:.4f}) | '
                f'val_loss={val_metrics["loss"]:.4f} '
                f'(pos={val_metrics["pos_loss"]:.4f}, conf={val_metrics["conf_loss"]:.4f})'
            )
            print(
                f'  distance_error={train_metrics["mean_distance_error"]:.4f} '
                f'top1_error={train_metrics["top1_enemy_position_error"]:.4f} '
                f'conf_prec={train_metrics["confidence_precision"]["known_enemy"]:.4f} '
                f'conf_recall={train_metrics["confidence_recall"]["known_enemy"]:.4f}'
            )
            print(f'  last_seen bucket error: {train_metrics["last_seen_bucket_distance_error"]}')

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
                        'pos_loss': train_metrics['pos_loss'],
                        'conf_loss': train_metrics['conf_loss'],
                        'mean_distance_error': train_metrics['mean_distance_error'],
                        'top1_enemy_position_error': train_metrics['top1_enemy_position_error'],
                        'confidence_precision': train_metrics['confidence_precision'],
                        'confidence_recall': train_metrics['confidence_recall'],
                        'confidence_f1': train_metrics['confidence_f1'],
                        'last_seen_bucket_distance_error': train_metrics['last_seen_bucket_distance_error'],
                        'coverage': train_coverage,
                    },
                    'val': {
                        'loss': val_metrics['loss'],
                        'pos_loss': val_metrics['pos_loss'],
                        'conf_loss': val_metrics['conf_loss'],
                        'mean_distance_error': val_metrics['mean_distance_error'],
                        'top1_enemy_position_error': val_metrics['top1_enemy_position_error'],
                        'confidence_precision': val_metrics['confidence_precision'],
                        'confidence_recall': val_metrics['confidence_recall'],
                        'confidence_f1': val_metrics['confidence_f1'],
                        'last_seen_bucket_distance_error': val_metrics['last_seen_bucket_distance_error'],
                        'coverage': val_coverage,
                    },
                },
            )

            log_scalar_dict(writer, 'train', train_metrics, epoch, ignored_keys=TRACKER_METRIC_DICT_KEYS)
            log_scalar_dict(writer, 'val', val_metrics, epoch, ignored_keys=TRACKER_METRIC_DICT_KEYS)
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
        best_train_metrics = {
            'loss': 0.0,
            'pos_loss': 0.0,
            'conf_loss': 0.0,
            'mean_distance_error': 0.0,
            'top1_enemy_position_error': 0.0,
            'confidence_precision': {'known_enemy': 0.0},
            'confidence_recall': {'known_enemy': 0.0},
            'confidence_f1': {'known_enemy': 0.0},
            'last_seen_bucket_distance_error': {},
        }
        best_val_metrics = dict(best_train_metrics)
    report = build_base_training_report(
        module_name='enemy_tracker',
        model_name=f'enemy_tracker_lstm_{args.output_mode}',
        dataset_path=dataset_label,
        split_mode=args.split_mode,
        seq_len=args.seq_len,
        feature_dim=feature_extractor.feature_dim(),
        target_shape='[batch, seq_len, enemies, 3]/[batch, seq_len, enemies] or target_tick variant',
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
