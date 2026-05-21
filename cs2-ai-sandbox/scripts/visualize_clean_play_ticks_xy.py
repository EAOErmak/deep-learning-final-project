from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd


POSITION_COLUMN_CANDIDATES = (
    ("X", "Y"),
    ("x", "y"),
    ("pos_x", "pos_y"),
    ("self_x", "self_y"),
    ("player_x", "player_y"),
)


def resolve_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    columns = set(df.columns)
    for x_col, y_col in POSITION_COLUMN_CANDIDATES:
        if x_col in columns and y_col in columns:
            return x_col, y_col
    raise ValueError(
        "Could not resolve X/Y columns. Expected one of: "
        "X/Y, x/y, pos_x/pos_y, self_x/self_y, player_x/player_y."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate player movement from one round parquet file.")
    parser.add_argument("--input", type=Path, default=None, help="Optional explicit path to one round parquet file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/rounds-dataset"),
        help="Root directory containing demo folders with rounds/*.parquet files.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*/rounds/*.parquet",
        help="Glob pattern under --input-dir used to find round parquet files.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic round selection.")
    parser.add_argument("--tick-step", type=int, default=1, help="Render every Nth tick.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Optional limit on rendered ticks.")
    parser.add_argument("--interval-ms", type=int, default=40, help="Animation interval in milliseconds.")
    parser.add_argument("--show-labels", action="store_true", help="Show player name labels next to points.")
    return parser.parse_args()


def build_tick_frames(df: pd.DataFrame, tick_step: int, max_ticks: int | None) -> list[tuple[int, pd.DataFrame]]:
    unique_ticks = sorted(int(tick) for tick in pd.unique(df["tick"]))
    if tick_step > 1:
        unique_ticks = unique_ticks[::tick_step]
    if max_ticks is not None:
        unique_ticks = unique_ticks[: max(0, int(max_ticks))]
    frames: list[tuple[int, pd.DataFrame]] = []
    for tick in unique_ticks:
        tick_rows = df.loc[df["tick"] == tick].copy()
        frames.append((tick, tick_rows))
    return frames


def main() -> int:
    args = parse_args()
    if args.input is not None:
        selected_path = args.input
    else:
        candidates = sorted(args.input_dir.glob(args.input_glob))
        if not candidates:
            raise FileNotFoundError(
                f"No round parquet files found for {args.input_dir} / {args.input_glob}"
            )
        rng = random.Random(args.seed)
        selected_path = rng.choice(candidates)
    if not selected_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {selected_path}")

    df = pd.read_parquet(selected_path)
    required_columns = {"tick", "steamid", "name", "team_num"}
    missing_required = sorted(column for column in required_columns if column not in df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    x_col, y_col = resolve_xy_columns(df)
    df = df.dropna(subset=[x_col, y_col, "tick", "steamid"]).copy()
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").astype("int64")
    df["steamid"] = pd.to_numeric(df["steamid"], errors="coerce").astype("int64")
    df["team_num"] = pd.to_numeric(df["team_num"], errors="coerce").fillna(0).astype("int64")
    df = df.sort_values(["tick", "team_num", "steamid"]).reset_index(drop=True)

    frames = build_tick_frames(df, tick_step=max(1, int(args.tick_step)), max_ticks=args.max_ticks)
    if not frames:
        raise ValueError("No frames to render after filtering.")

    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    y_min = float(df[y_col].min())
    y_max = float(df[y_col].max())
    x_margin = max((x_max - x_min) * 0.05, 50.0)
    y_margin = max((y_max - y_min) * 0.05, 50.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(selected_path.name)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    team_colors = {
        2: "#d94f30",  # T
        3: "#3b82f6",  # CT
    }
    unknown_color = "#9ca3af"

    scatter = ax.scatter([], [], s=70, c=[])
    title_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    label_artists: list[plt.Text] = []

    def update(frame_index: int):
        nonlocal label_artists
        tick_value, tick_rows = frames[frame_index]
        offsets = tick_rows[[x_col, y_col]].to_numpy(dtype=float)
        colors = [team_colors.get(int(team_num), unknown_color) for team_num in tick_rows["team_num"].tolist()]
        scatter.set_offsets(offsets)
        scatter.set_color(colors)

        for artist in label_artists:
            artist.remove()
        label_artists = []
        if args.show_labels:
            for row in tick_rows.itertuples(index=False):
                label_artists.append(
                    ax.text(
                        float(getattr(row, x_col)),
                        float(getattr(row, y_col)),
                        str(getattr(row, "name")),
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )
                )

        alive_count = int(tick_rows["is_alive"].fillna(False).astype(bool).sum()) if "is_alive" in tick_rows.columns else len(tick_rows)
        title_text.set_text(f"tick={tick_value} players={len(tick_rows)} alive={alive_count}")
        return [scatter, title_text, *label_artists]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(args.interval_ms)),
        blit=False,
        repeat=True,
    )
    update(0)

    print(f"Loaded: {selected_path}")
    print(f"Rows: {len(df)} | Unique ticks: {df['tick'].nunique()} | Rendered frames: {len(frames)}")
    print(f"Players: {df['steamid'].nunique()} | X/Y columns: {x_col}/{y_col}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
