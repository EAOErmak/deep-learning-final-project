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
from dataclasses import dataclass
from pathlib import Path

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from cs2_ai.dataset.parquet_loader import load_clean_buy_tick_files, load_parquet, parquet_demo_name
from cs2_ai.features.buy_features import BuyFeatureExtractor, build_buy_target_from_freeze_sequence

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

BUY_LABELS = ['awp', 'eco', 'force', 'full', 'half']
BUY_LABEL_TO_ID = {label: idx for idx, label in enumerate(BUY_LABELS)}


@dataclass(slots=True)
class BuySample:
    features: np.ndarray
    buy_type: str
    money_spent: float
    demo_name: str
    round_number: int
    steamid: int
    sample_id: str


class BuySampleBuilder:
    def __init__(self, dataset_dir: Path, show_progress: bool = False, max_demos: int | None = None, max_samples: int | None = None):
        self.dataset_dir = dataset_dir
        self.show_progress = show_progress
        self.max_demos = max_demos
        self.max_samples = max_samples
        self.extractor = BuyFeatureExtractor()

    def build(self) -> list[BuySample]:
        files = load_clean_buy_tick_files(self.dataset_dir)
        if self.max_demos is not None:
            files = files[: self.max_demos]
        if not files:
            raise FileNotFoundError(f'No clean_buy_ticks parquet found in {self.dataset_dir / "clean_buy_ticks"}')

        iterator = files
        if self.show_progress and tqdm is not None:
            iterator = tqdm(files, desc='Building buy samples', unit='demo')

        samples: list[BuySample] = []
        for parquet_path in iterator:
            demo_name = parquet_demo_name(parquet_path)
            df = load_parquet(parquet_path)
            if df.empty:
                continue
            samples.extend(self._build_demo_samples(df, demo_name))
            if self.max_samples is not None and len(samples) >= self.max_samples:
                return samples[: self.max_samples]
        return samples

    def _build_demo_samples(self, df: pd.DataFrame, demo_name: str) -> list[BuySample]:
        if 'tick' not in df.columns or 'steamid' not in df.columns or 'total_rounds_played' not in df.columns:
            return []
        samples: list[BuySample] = []
        round_groups = df.groupby('total_rounds_played', sort=True)
        for round_number, round_df in round_groups:
            if round_df.empty:
                continue
            first_tick = int(round_df['tick'].min())
            first_tick_rows = round_df.loc[round_df['tick'] == first_tick].copy()
            if first_tick_rows.empty:
                continue
            steamids = pd.to_numeric(first_tick_rows['steamid'], errors='coerce').dropna().astype('int64').unique().tolist()
            for steamid in sorted(int(value) for value in steamids):
                player_rows = round_df.loc[pd.to_numeric(round_df['steamid'], errors='coerce') == steamid].copy()
                if player_rows.empty:
                    continue
                try:
                    features = self.extractor.extract_from_tick_rows(first_tick_rows, steamid)
                except Exception:
                    continue
                target = build_buy_target_from_freeze_sequence(player_rows)
                buy_type = str(target['buy_type'])
                if buy_type not in BUY_LABEL_TO_ID:
                    continue
                sample_id = f'{demo_name}|r{int(round_number)}|p{steamid}'
                samples.append(
                    BuySample(
                        features=features.astype(np.float32),
                        buy_type=buy_type,
                        money_spent=float(target['money_spent']),
                        demo_name=demo_name,
                        round_number=int(round_number),
                        steamid=steamid,
                        sample_id=sample_id,
                    )
                )
        return samples


def split_buy_samples(samples: list[BuySample], val_split: float, split_mode: str, seed: int) -> tuple[list[int], list[int]]:
    if len(samples) < 2:
        raise ValueError('Need at least 2 buy samples for train/val split.')

    if split_mode == 'random':
        indices = list(range(len(samples)))
        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=seed, shuffle=True)
        return sorted(train_idx), sorted(val_idx)

    if split_mode == 'demo':
        groups = [sample.demo_name for sample in samples]
    elif split_mode == 'round':
        groups = [f'{sample.demo_name}|r{sample.round_number}' for sample in samples]
    else:
        raise ValueError(f'Unsupported split_mode: {split_mode}')

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError(f'split_mode={split_mode} requires at least 2 unique groups, got {len(unique_groups)}')
    train_groups, val_groups = train_test_split(unique_groups, test_size=val_split, random_state=seed, shuffle=True)
    train_group_set = set(train_groups)
    val_group_set = set(val_groups)
    train_idx = [idx for idx, group in enumerate(groups) if group in train_group_set]
    val_idx = [idx for idx, group in enumerate(groups) if group in val_group_set]
    if not train_idx or not val_idx:
        raise ValueError('Grouped split produced an empty train or val split.')
    return train_idx, val_idx


def materialize_xy(samples: list[BuySample], indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([samples[idx].features for idx in indices]).astype(np.float32)
    y_type = np.asarray([BUY_LABEL_TO_ID[samples[idx].buy_type] for idx in indices], dtype=np.int64)
    y_spent = np.asarray([samples[idx].money_spent for idx in indices], dtype=np.float32)
    return x, y_type, y_spent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a classical ML buy model from clean_buy_ticks')
    parser.add_argument('--dataset-dir', type=Path, default=PROJECT_ROOT / 'dataset')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--split-mode', choices=['demo', 'round', 'random'], default='demo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--max-depth', type=int, default=12)
    parser.add_argument('--max-demos', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--show-build-progress', action='store_true')
    parser.add_argument('--save-path', type=Path, default=PROJECT_ROOT / 'checkpoints' / 'buy_sklearn.joblib')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = BuySampleBuilder(
        dataset_dir=args.dataset_dir,
        show_progress=args.show_build_progress,
        max_demos=args.max_demos,
        max_samples=args.max_samples,
    )
    samples = builder.build()
    if not samples:
        raise FileNotFoundError(f'No buy samples could be built from {args.dataset_dir / "clean_buy_ticks"}')

    print('train_buy_sklearn.py')
    print(f'Total buy samples: {len(samples)}')
    print(f'Feature dim: {builder.extractor.feature_dim()}')
    print(f'Split mode: {args.split_mode}')

    train_idx, val_idx = split_buy_samples(samples, args.val_split, args.split_mode, args.seed)
    print(f'Train samples: {len(train_idx)}')
    print(f'Val samples: {len(val_idx)}')

    x_train, y_train_type, y_train_spent = materialize_xy(samples, train_idx)
    x_val, y_val_type, y_val_spent = materialize_xy(samples, val_idx)

    train_class_ids = sorted(set(int(value) for value in y_train_type.tolist()))
    missing_train_classes = [BUY_LABELS[class_id] for class_id in range(len(BUY_LABELS)) if class_id not in train_class_ids]
    if missing_train_classes:
        print(f'Warning: train split is missing classes: {missing_train_classes}')

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
        class_weight='balanced_subsample',
    )
    regressor = GradientBoostingRegressor(random_state=args.seed)

    print('Training buy_type classifier...')
    classifier.fit(x_train, y_train_type)
    print('Training money_spent regressor...')
    regressor.fit(x_train, y_train_spent)

    train_pred_type = classifier.predict(x_train)
    val_pred_type = classifier.predict(x_val)
    train_pred_spent = regressor.predict(x_train)
    val_pred_spent = regressor.predict(x_val)

    train_metrics = {
        'buy_type_accuracy': float(accuracy_score(y_train_type, train_pred_type)),
        'buy_type_balanced_accuracy': float(balanced_accuracy_score(y_train_type, train_pred_type)),
        'money_spent_rmse': float(math.sqrt(mean_squared_error(y_train_spent, train_pred_spent))),
        'money_spent_mae': float(mean_absolute_error(y_train_spent, train_pred_spent)),
    }
    val_metrics = {
        'buy_type_accuracy': float(accuracy_score(y_val_type, val_pred_type)),
        'buy_type_balanced_accuracy': float(balanced_accuracy_score(y_val_type, val_pred_type)),
        'money_spent_rmse': float(math.sqrt(mean_squared_error(y_val_spent, val_pred_spent))),
        'money_spent_mae': float(mean_absolute_error(y_val_spent, val_pred_spent)),
    }

    confusion = confusion_matrix(y_val_type, val_pred_type, labels=list(range(len(BUY_LABELS))))

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        'model_type': 'buy_sklearn_baseline',
        'classifier': classifier,
        'regressor': regressor,
        'feature_names': builder.extractor.feature_names(),
        'label_classes': BUY_LABELS,
        'label_to_id': BUY_LABEL_TO_ID,
        'config': {
            'dataset_dir': str(args.dataset_dir),
            'val_split': args.val_split,
            'split_mode': args.split_mode,
            'seed': args.seed,
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'max_demos': args.max_demos,
            'max_samples': args.max_samples,
        },
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_present_classes': [BUY_LABELS[class_id] for class_id in train_class_ids],
        'missing_train_classes': missing_train_classes,
        'val_confusion_matrix': confusion.tolist(),
    }
    dump(artifact, args.save_path)

    metrics_path = args.save_path.with_suffix('.metrics.json')
    metrics_path.write_text(
        json.dumps(
            {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'label_classes': BUY_LABELS,
                'train_present_classes': [BUY_LABELS[class_id] for class_id in train_class_ids],
                'missing_train_classes': missing_train_classes,
                'val_confusion_matrix': confusion.tolist(),
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print('Training finished.')
    print(f'Saved model: {args.save_path}')
    print(f'Saved metrics: {metrics_path}')
    print(f'Train metrics: {train_metrics}')
    print(f'Val metrics: {val_metrics}')
    print('Label classes:', BUY_LABELS)
    print('Train present classes:', [BUY_LABELS[class_id] for class_id in train_class_ids])
    if missing_train_classes:
        print('Missing train classes:', missing_train_classes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())