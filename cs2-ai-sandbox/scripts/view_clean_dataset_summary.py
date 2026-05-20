from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

from pathlib import Path

import pandas as pd


def list_parquet_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def print_shapes(title: str, files: list[Path]) -> None:
    print(f"\n== {title} ==")
    if not files:
        print("No parquet files found.")
        return

    for path in files:
        try:
            df = pd.read_parquet(path)
            print(f"{path.name}: shape={df.shape}")
        except Exception as exc:
            print(f"{path.name}: failed to read ({exc})")


def print_preview(title: str, files: list[Path]) -> None:
    print(f"\n== {title} Preview ==")
    if not files:
        print("No parquet files found.")
        return

    try:
        df = pd.read_parquet(files[0])
        print(df.head())
    except Exception as exc:
        print(f"Failed to preview {files[0].name}: {exc}")


def rate(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    series = df[column]
    if series.dtype == bool:
        return float(series.mean())
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float((numeric != 0).mean())


def mouse_nonzero_rate(df: pd.DataFrame) -> float:
    if not {"usercmd_mouse_dx", "usercmd_mouse_dy"}.issubset(df.columns) or df.empty:
        return 0.0
    mouse_dx = pd.to_numeric(df["usercmd_mouse_dx"], errors="coerce").fillna(0.0)
    mouse_dy = pd.to_numeric(df["usercmd_mouse_dy"], errors="coerce").fillna(0.0)
    return float(((mouse_dx != 0.0) | (mouse_dy != 0.0)).mean())


def print_dataset_metrics(title: str, files: list[Path]) -> None:
    print(f"\n== {title} Metrics ==")
    if not files:
        print("No parquet files found.")
        return

    for path in files[:10]:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            print(f"{path.name}: failed to read ({exc})")
            continue

        metrics: list[str] = []
        if title == "clean_play_ticks":
            metrics.append(f"spotted={rate(df, 'spotted'):.2%}")
            metrics.append(f"fire={rate(df, 'FIRE'):.2%}")
            metrics.append(f"mouse_nonzero={mouse_nonzero_rate(df):.2%}")
            if "round_in_progress" in df.columns:
                metrics.append(f"round_in_progress={rate(df, 'round_in_progress'):.2%}")
            if "is_freeze_period" in df.columns:
                metrics.append(f"freeze_leak={rate(df, 'is_freeze_period'):.2%}")
            if "is_warmup_period" in df.columns:
                metrics.append(f"warmup_leak={rate(df, 'is_warmup_period'):.2%}")
        elif title == "clean_buy_ticks":
            if "in_buy_zone" in df.columns:
                metrics.append(f"in_buy_zone={rate(df, 'in_buy_zone'):.2%}")
            if "is_freeze_period" in df.columns:
                metrics.append(f"freeze={rate(df, 'is_freeze_period'):.2%}")
        elif title == "round_events":
            if "winner_team_num_known" in df.columns:
                metrics.append(f"winner_known={rate(df, 'winner_team_num_known'):.2%}")
            if "reward_known" in df.columns:
                metrics.append(f"reward_known={rate(df, 'reward_known'):.2%}")

        print(f"{path.name}: shape={df.shape} | " + ", ".join(metrics))


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"

    raw_files = list_parquet_files(dataset_dir / "raw_ticks")
    play_files = list_parquet_files(dataset_dir / "clean_play_ticks")
    buy_files = list_parquet_files(dataset_dir / "clean_buy_ticks")
    round_files = list_parquet_files(dataset_dir / "round_events")

    print_shapes("raw_ticks", raw_files)
    print_shapes("clean_play_ticks", play_files)
    print_shapes("clean_buy_ticks", buy_files)
    print_shapes("round_events", round_files)
    print_dataset_metrics("clean_play_ticks", play_files)
    print_dataset_metrics("clean_buy_ticks", buy_files)
    print_dataset_metrics("round_events", round_files)

    print_preview("clean_play_ticks", play_files)
    print_preview("clean_buy_ticks", buy_files)
    print_preview("round_events", round_files)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
