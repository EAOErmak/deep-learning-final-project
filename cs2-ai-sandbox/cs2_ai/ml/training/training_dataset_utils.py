from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    from torch.utils.data import Subset
except Exception:
    class Subset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

from cs2_ai.dataset.round_identity import round_metadata_from_sample
from cs2_ai.ml.training.training_ledger import TrainingRoundLedger


def add_common_training_data_args(parser: argparse.ArgumentParser, *, project_root: Path, legacy_dataset_dir: bool = True) -> None:
    parser.add_argument('--data-dir', type=Path, default=project_root / 'data')
    parser.add_argument('--dataset-subdir', type=str, default='rounds-dataset')
    if legacy_dataset_dir:
        parser.add_argument('--dataset-dir', type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--rounds-ledger-path', type=Path, default=project_root / 'artifacts' / 'training_ledger' / 'training_rounds.jsonl')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--skip-trained-rounds', action='store_true')
    parser.add_argument('--ledger-match-mode', choices=['module', 'model', 'checkpoint', 'any'], default='module')


def resolve_dataset_root(args: argparse.Namespace, project_root: Path) -> Path:
    if getattr(args, 'dataset_dir', None) is not None:
        return Path(args.dataset_dir)
    candidate = Path(args.data_dir)
    legacy = project_root / 'dataset'
    if candidate.exists():
        return candidate
    if legacy.exists():
        return legacy
    return candidate


def build_dataset_label(args: argparse.Namespace, project_root: Path) -> str:
    return str(resolve_dataset_root(args, project_root) / args.dataset_subdir)


def resolve_run_id(args: argparse.Namespace, module_name: str) -> str:
    if getattr(args, 'run_id', None):
        return str(args.run_id)
    return f'{module_name}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'


def filter_dataset_by_trained_rounds(
    dataset,
    *,
    ledger_path: Path,
    module_name: str,
    model_name: str,
    checkpoint_path: str,
    match_mode: str,
):
    ledger = TrainingRoundLedger.load(ledger_path)
    trained_round_uids = ledger.read_trained_round_uids(
        module_name=module_name,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        match_mode=match_mode,
    )
    round_uids_before = {str(round_metadata_from_sample(dataset.get_sample_metadata(idx))['round_uid']) for idx in range(len(dataset))}
    keep_indices: list[int] = []
    for idx in range(len(dataset)):
        round_uid = str(round_metadata_from_sample(dataset.get_sample_metadata(idx))['round_uid'])
        if round_uid not in trained_round_uids:
            keep_indices.append(idx)
    remaining_round_uids = {str(round_metadata_from_sample(dataset.get_sample_metadata(idx))['round_uid']) for idx in keep_indices}
    skipped_round_count = len(round_uids_before - remaining_round_uids)
    filtered = Subset(dataset, keep_indices)
    return filtered, {
        'total_rounds_before_skip': len(round_uids_before),
        'skipped_rounds_count': skipped_round_count,
        'remaining_rounds_count': len(remaining_round_uids),
        'trained_round_uids': trained_round_uids,
    }
