from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.ml.utils.torch_utils import get_device, torch_available


def main() -> int:
    if not torch_available():
        print("PyTorch is not available. Install torch to use train_movement.py")
        return 0
    print("train_movement.py")
    print(f"Device: {get_device()}")
    print("TODO: use MovementFeatureExtractor")
    print("TODO: build movement targets")
    print("Training loop is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
