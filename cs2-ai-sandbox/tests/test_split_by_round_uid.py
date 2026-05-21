from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset, split_dataset_by_group
from tests.test_rounds_dataset_loader import make_rows


class SplitByRoundUidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = PROJECT_ROOT / '.tmp_test_split_round_uid'
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        for demo_name in ('demo_a_play_ticks', 'demo_b_play_ticks'):
            rounds_dir = self.tmp_root / 'data' / 'rounds-dataset' / demo_name / 'rounds'
            rounds_dir.mkdir(parents=True, exist_ok=True)
            make_rows(1, (1, 2)).to_parquet(rounds_dir / 'round_1.parquet', index=False)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_demo_round_one_groups_are_distinct(self):
        dataset = MultiDemoSequenceDataset(dataset_dir=self.tmp_root / 'data', subdir='rounds-dataset', seq_len=2, stride=1)
        round_uids = {dataset.get_sample_metadata(idx)['round_uid'] for idx in range(len(dataset))}
        self.assertIn('demo_a_play_ticks::round_1', round_uids)
        self.assertIn('demo_b_play_ticks::round_1', round_uids)
        self.assertEqual(len(round_uids), 2)

    def test_split_mode_round_does_not_leak_same_round_uid(self):
        dataset = MultiDemoSequenceDataset(dataset_dir=self.tmp_root / 'data', subdir='rounds-dataset', seq_len=2, stride=1)
        train_subset, val_subset = split_dataset_by_group(dataset, val_split=0.5, seed=42, mode='round')
        train_rounds = {dataset.get_sample_metadata(idx)['round_uid'] for idx in train_subset.indices}
        val_rounds = {dataset.get_sample_metadata(idx)['round_uid'] for idx in val_subset.indices}
        self.assertTrue(train_rounds.isdisjoint(val_rounds))


if __name__ == '__main__':
    unittest.main()
