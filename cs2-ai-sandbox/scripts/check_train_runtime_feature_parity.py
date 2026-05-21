from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset
from cs2_ai.features.aim_features import AIM_FEATURE_MODE_DEMO_PROJECTED, AIM_FEATURE_MODE_VISION_LIKE, AimFeatureExtractor
from cs2_ai.features.enemy_tracker_features import EnemyTrackerFeatureExtractor
from cs2_ai.features.feature_contract import validate_checkpoint_schema
from cs2_ai.features.movement_features import MovementFeatureExtractor

try:
    import torch
except Exception:
    torch = None


def _compare_arrays(name: str, lhs: np.ndarray, rhs: np.ndarray) -> list[str]:
    issues: list[str] = []
    if lhs.shape != rhs.shape:
        issues.append(f"{name}: shape mismatch {lhs.shape} != {rhs.shape}")
        return issues
    if not np.allclose(lhs, rhs, atol=1e-6):
        diff_index = tuple(int(v) for v in np.argwhere(np.abs(lhs - rhs) > 1e-6)[0].tolist())
        issues.append(
            f"{name}: first differing value at {diff_index}: offline={lhs[diff_index]!r} runtime={rhs[diff_index]!r}"
        )
    return issues


def _load_checkpoint(path: str | None) -> dict[str, object] | None:
    if path is None:
        return None
    if torch is None:
        raise RuntimeError("PyTorch is required to validate checkpoint metadata.")
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare offline and runtime-style feature extraction for one sample.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--aim-feature-mode", choices=[AIM_FEATURE_MODE_DEMO_PROJECTED, AIM_FEATURE_MODE_VISION_LIKE], default=AIM_FEATURE_MODE_DEMO_PROJECTED)
    parser.add_argument("--aim-checkpoint", type=str, default=None)
    parser.add_argument("--movement-checkpoint", type=str, default=None)
    parser.add_argument("--tracker-checkpoint", type=str, default=None)
    args = parser.parse_args()

    dataset = MultiDemoSequenceDataset(dataset_dir=args.dataset_dir, subdir="clean_play_ticks", seq_len=args.seq_len, stride=4, alive_only=True, max_samples_total=max(args.sample_index + 1, 1))
    if len(dataset) <= args.sample_index:
        print("FAIL")
        print(f"sample-index {args.sample_index} is out of range for dataset size {len(dataset)}")
        return 1

    sample = dataset[args.sample_index]
    tracker_offline = EnemyTrackerFeatureExtractor(seq_len=args.seq_len)
    tracker_runtime = EnemyTrackerFeatureExtractor(seq_len=args.seq_len)
    movement_offline = MovementFeatureExtractor(seq_len=args.seq_len)
    movement_runtime = MovementFeatureExtractor(seq_len=args.seq_len)
    aim_offline = AimFeatureExtractor(seq_len=args.seq_len, feature_mode=args.aim_feature_mode)
    aim_runtime = AimFeatureExtractor(seq_len=args.seq_len, feature_mode=args.aim_feature_mode)

    checks: list[str] = []
    schemas = {
        "tracker": tracker_offline.schema(),
        "movement": movement_offline.schema(),
        "aim": aim_offline.schema(),
    }

    comparisons = {
        "tracker": (tracker_offline.extract(sample.sequence), tracker_runtime.extract(sample.sequence)),
        "movement": (movement_offline.extract(sample.sequence), movement_runtime.extract(sample.sequence)),
        "aim": (aim_offline.extract(sample.sequence), aim_runtime.extract(sample.sequence)),
    }

    for name, (offline_features, runtime_features) in comparisons.items():
        schema = schemas[name]
        checks.extend(_compare_arrays(name, offline_features, runtime_features))
        if offline_features.shape != (args.seq_len, schema.feature_dim):
            checks.append(f"{name}: expected shape {(args.seq_len, schema.feature_dim)} got {offline_features.shape}")
        if len(schema.feature_names) != schema.feature_dim:
            checks.append(f"{name}: feature_names len mismatch {len(schema.feature_names)} != {schema.feature_dim}")

    checkpoints = {
        "aim": _load_checkpoint(args.aim_checkpoint),
        "movement": _load_checkpoint(args.movement_checkpoint),
        "tracker": _load_checkpoint(args.tracker_checkpoint),
    }
    for name, checkpoint in checkpoints.items():
        if checkpoint is None:
            continue
        try:
            validate_checkpoint_schema(checkpoint, schemas[name], args.__dict__[f"{name}_checkpoint"])
        except Exception as exc:
            checks.append(f"{name}: checkpoint validation failed: {exc}")

    if checks:
        print("FAIL")
        for issue in checks:
            print(issue)
        return 1

    print("PASS")
    for name, schema in schemas.items():
        print(
            f"{name}: feature_dim={schema.feature_dim} seq_len={schema.seq_len} "
            f"schema_hash={schema.schema_hash}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
