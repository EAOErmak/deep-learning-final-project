from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.ml.training.training_ledger import TrainingRoundLedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect training round ledger.')
    parser.add_argument('--ledger-path', type=Path, default=Path('artifacts/training_ledger/training_rounds.jsonl'))
    parser.add_argument('--module', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger = TrainingRoundLedger.load(args.ledger_path)
    entries = ledger.entries
    if args.module is not None:
        entries = [entry for entry in entries if str(entry.get('module_name')) == args.module]
    if args.model is not None:
        entries = [entry for entry in entries if str(entry.get('model_name')) == args.model]
    if args.checkpoint is not None:
        entries = [entry for entry in entries if str(entry.get('checkpoint_path')) == args.checkpoint]

    runs = sorted({str(entry['run_id']) for entry in entries})
    unique_rounds = {str(entry['round_uid']) for entry in entries}
    rounds_per_module: dict[str, int] = {}
    rounds_per_model: dict[str, int] = {}
    checkpoints_seen = sorted({str(entry['checkpoint_path']) for entry in entries})
    repeated_rounds_count = len(entries) - len(unique_rounds)
    for entry in entries:
        rounds_per_module[str(entry['module_name'])] = rounds_per_module.get(str(entry['module_name']), 0) + 1
        rounds_per_model[str(entry['model_name'])] = rounds_per_model.get(str(entry['model_name']), 0) + 1

    print(f'total runs: {len(runs)}')
    print(f'total unique rounds trained: {len(unique_rounds)}')
    print(f'rounds per module: {rounds_per_module}')
    print(f'rounds per model: {rounds_per_model}')
    print(f'repeated rounds count: {repeated_rounds_count}')
    print(f'checkpoints seen: {checkpoints_seen}')
    print('last 10 runs:')
    for run_id in runs[-10:]:
        print(f'  {run_id}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
