from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM
from cs2_ai.ml.utils.torch_utils import get_device, torch_available


def main() -> int:
    if not torch_available():
        print("PyTorch is not available. Install torch to use train_enemy_tracker.py")
        return 0
    print("train_enemy_tracker.py")
    print(f"Device: {get_device()}")
    print("TODO: load PerspectiveSequenceDataset")
    print("TODO: use EnemyTrackerFeatureExtractor")
    print("TODO: build enemy position targets")
    print("TODO: train EnemyTrackerLSTM")
    _ = EnemyTrackerLSTM(input_dim=32)
    print("Training loop is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
