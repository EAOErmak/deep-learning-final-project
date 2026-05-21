from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.ml.training.training_dataset_utils import filter_dataset_by_trained_rounds
from cs2_ai.ml.training.training_ledger import TrainingRoundLedger, collect_round_usage


class FakeDataset:
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def get_sample_metadata(self, idx):
        return dict(self.items[idx])


class TrainingLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = PROJECT_ROOT / f'.tmp_test_training_ledger_{uuid.uuid4().hex}'
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.tmp_root / 'ledger.jsonl'
        self.dataset = FakeDataset(
            [
                {
                    'sample_id': 'a',
                    'demo_name': 'demo_a',
                    'demo_dir': 'demo_a',
                    'round_number': 1,
                    'round_file': 'round_1.parquet',
                    'source_file': 'round_1.parquet',
                    'round_uid': 'demo_a::round_1',
                    'perspective_steamid': 1,
                    'tick_indices': (1, 2, 3),
                    'target_tick': 4,
                },
                {
                    'sample_id': 'b',
                    'demo_name': 'demo_b',
                    'demo_dir': 'demo_b',
                    'round_number': 1,
                    'round_file': 'round_1.parquet',
                    'source_file': 'round_1.parquet',
                    'round_uid': 'demo_b::round_1',
                    'perspective_steamid': 2,
                    'tick_indices': (1, 2, 3),
                    'target_tick': 4,
                },
            ]
        )

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_append_and_read_round_uids(self):
        ledger = TrainingRoundLedger.load(self.ledger_path)
        usage = collect_round_usage(self.dataset)
        ledger.append_run_rounds(
            run_id='run_1',
            module_name='movement',
            model_name='movement_gru',
            checkpoint_path='checkpoints/a.pt',
            dataset_dir='data',
            dataset_subdir='rounds-dataset',
            split_mode='round',
            split='train',
            round_usage=usage,
        )
        loaded = TrainingRoundLedger.load(self.ledger_path)
        round_uids = loaded.read_trained_round_uids(module_name='movement', match_mode='module')
        self.assertEqual(round_uids, {'demo_a::round_1', 'demo_b::round_1'})

    def test_collect_round_usage(self):
        usage = collect_round_usage(self.dataset)
        self.assertEqual(len(usage), 2)
        self.assertEqual(usage[0]['sample_count'], 1)

    def test_skip_trained_rounds_removes_expected_rounds(self):
        ledger = TrainingRoundLedger.load(self.ledger_path)
        ledger.append_run_rounds(
            run_id='run_1',
            module_name='movement',
            model_name='movement_gru',
            checkpoint_path='checkpoints/a.pt',
            dataset_dir='data',
            dataset_subdir='rounds-dataset',
            split_mode='round',
            split='train',
            round_usage=[collect_round_usage(self.dataset)[0]],
        )
        filtered, info = filter_dataset_by_trained_rounds(
            self.dataset,
            ledger_path=self.ledger_path,
            module_name='movement',
            model_name='movement_gru',
            checkpoint_path='checkpoints/a.pt',
            match_mode='module',
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(info['skipped_rounds_count'], 1)


if __name__ == '__main__':
    unittest.main()
