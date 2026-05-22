from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POSITION_COLUMN_CANDIDATES = (
    ("X", "Y", "Z"),
    ("x", "y", "z"),
    ("pos_x", "pos_y", "pos_z"),
    ("self_x", "self_y", "self_z"),
    ("player_x", "player_y", "player_z"),
)


@dataclass(slots=True)
class PlayerTrack3D:
    steamid: int
    name: str
    team_num: int
    row_to_segment: np.ndarray
    segment_centers_xyz: np.ndarray

    def future_cells_xyz(self, row_track_index: int, future_cells: int) -> np.ndarray:
        if future_cells <= 0:
            return np.empty((0, 3), dtype=float)
        segment_index = int(self.row_to_segment[int(row_track_index)])
        future_start = segment_index + 1
        future_end = min(future_start + int(future_cells), len(self.segment_centers_xyz))
        if future_start >= future_end:
            return np.empty((0, 3), dtype=float)
        return self.segment_centers_xyz[future_start:future_end]


def resolve_xyz_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    columns = set(df.columns)
    for x_col, y_col, z_col in POSITION_COLUMN_CANDIDATES:
        if x_col in columns and y_col in columns and z_col in columns:
            return x_col, y_col, z_col
    raise ValueError(
        "Could not resolve X/Y/Z columns. Expected one of: "
        "X/Y/Z, x/y/z, pos_x/pos_y/pos_z, self_x/self_y/self_z, player_x/player_y/player_z."
    )


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate one grid-labeled round parquet file in 3D and overlay future grid-cell centers per player."
    )
    parser.add_argument("--input", type=Path, default=None, help="Optional explicit path to one grid-labeled round parquet file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/rounds-dataset-grid"),
        help="Root directory containing demo folders with rounds/*.parquet files.",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*/rounds/round_*.parquet",
        help="Glob pattern under --input-dir used to find round parquet files.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for deterministic round selection.")
    parser.add_argument("--tick-step", type=int, default=2, help="Render every Nth tick.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Optional limit on rendered ticks.")
    parser.add_argument("--ticks-per-second", type=float, default=12.0, help="Playback speed in rendered ticks per second.")
    parser.add_argument("--future-cells", type=int, default=20, help="How many future grid cells to highlight per player.")
    parser.add_argument("--show-labels", action="store_true", help="Show player name labels next to points.")
    parser.add_argument("--elev", type=float, default=35.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-60.0, help="3D camera azimuth.")
    return parser.parse_args()


def resolve_interval_ms(ticks_per_second: float, tick_step: int) -> int:
    if ticks_per_second <= 0:
        raise ValueError(f"--ticks-per-second must be > 0, got {ticks_per_second}")
    return max(1, int(round((1000.0 * max(1, int(tick_step))) / float(ticks_per_second))))


def build_tick_frames(df: pd.DataFrame, tick_step: int, max_ticks: int | None) -> list[tuple[int, pd.DataFrame]]:
    unique_ticks = sorted(int(tick) for tick in pd.unique(df["tick"]))
    if tick_step > 1:
        unique_ticks = unique_ticks[::tick_step]
    if max_ticks is not None:
        unique_ticks = unique_ticks[: max(0, int(max_ticks))]
    return [(tick, df.loc[df["tick"] == tick].copy()) for tick in unique_ticks]


def select_round_file(args: argparse.Namespace) -> Path:
    if args.input is not None:
        return args.input
    candidates = sorted(args.input_dir.glob(args.input_glob))
    if not candidates:
        raise FileNotFoundError(f"No round parquet files found for {args.input_dir} / {args.input_glob}")
    return random.Random(args.seed).choice(candidates)


def load_round_dataframe(path: Path) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    df = pd.read_parquet(path)
    require_columns(
        df,
        [
            "tick",
            "steamid",
            "name",
            "team_num",
            "current_cell_id",
            "cell_center_x",
            "cell_center_y",
            "cell_center_z",
        ],
    )
    x_col, y_col, z_col = resolve_xyz_columns(df)
    df = df.dropna(
        subset=[
            x_col,
            y_col,
            z_col,
            "tick",
            "steamid",
            "current_cell_id",
            "cell_center_x",
            "cell_center_y",
            "cell_center_z",
        ]
    ).copy()
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").astype("int64")
    df["steamid"] = pd.to_numeric(df["steamid"], errors="coerce").astype("int64")
    df["team_num"] = pd.to_numeric(df["team_num"], errors="coerce").fillna(0).astype("int64")
    df["current_cell_id"] = pd.to_numeric(df["current_cell_id"], errors="coerce").astype("int64")
    df["cell_center_x"] = pd.to_numeric(df["cell_center_x"], errors="coerce")
    df["cell_center_y"] = pd.to_numeric(df["cell_center_y"], errors="coerce")
    df["cell_center_z"] = pd.to_numeric(df["cell_center_z"], errors="coerce")
    df = df.sort_values(["steamid", "tick"]).reset_index(drop=True)
    df["player_track_index"] = df.groupby("steamid", sort=False).cumcount().astype("int64")
    return df, (x_col, y_col, z_col)


def build_player_tracks(df: pd.DataFrame) -> dict[int, PlayerTrack3D]:
    tracks: dict[int, PlayerTrack3D] = {}
    for steamid, group in df.groupby("steamid", sort=False, dropna=False):
        ordered = group.sort_values(["tick", "player_track_index"]).reset_index(drop=True)
        cell_ids = pd.to_numeric(ordered["current_cell_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        centers_xyz = ordered[["cell_center_x", "cell_center_y", "cell_center_z"]].to_numpy(dtype=float, copy=False)
        segment_start_mask = np.ones(len(ordered), dtype=bool)
        if len(ordered) > 1:
            segment_start_mask[1:] = cell_ids[1:] != cell_ids[:-1]
        segment_start_indices = np.flatnonzero(segment_start_mask).astype(np.int64)
        row_to_segment = np.empty(len(ordered), dtype=np.int64)
        next_starts = np.append(segment_start_indices[1:], len(ordered))
        for segment_idx, (start_idx, end_idx) in enumerate(zip(segment_start_indices, next_starts, strict=True)):
            row_to_segment[start_idx:end_idx] = segment_idx
        tracks[int(steamid)] = PlayerTrack3D(
            steamid=int(steamid),
            name=str(ordered["name"].iloc[0]),
            team_num=int(pd.to_numeric(ordered["team_num"], errors="coerce").fillna(0).iloc[0]),
            row_to_segment=row_to_segment,
            segment_centers_xyz=centers_xyz[segment_start_indices],
        )
    return tracks


def set_scatter_xyz(scatter, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        scatter._offsets3d = ([], [], [])
        return
    scatter._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])


def main() -> int:
    args = parse_args()
    interval_ms = resolve_interval_ms(float(args.ticks_per_second), int(args.tick_step))
    selected_path = select_round_file(args)
    if not selected_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {selected_path}")

    df, (x_col, y_col, z_col) = load_round_dataframe(selected_path)
    frames = build_tick_frames(df, tick_step=max(1, int(args.tick_step)), max_ticks=args.max_ticks)
    if not frames:
        raise ValueError("No frames to render after filtering.")

    player_tracks = build_player_tracks(df)
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=float(args.elev), azim=float(args.azim))

    x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
    y_min, y_max = float(df[y_col].min()), float(df[y_col].max())
    z_min, z_max = float(df[z_col].min()), float(df[z_col].max())
    x_margin = max((x_max - x_min) * 0.05, 50.0)
    y_margin = max((y_max - y_min) * 0.05, 50.0)
    z_margin = max((z_max - z_min) * 0.05, 16.0)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(selected_path.name)

    team_colors = {2: "#d94f30", 3: "#3b82f6"}
    unknown_color = "#9ca3af"

    players_scatter = ax.scatter([], [], [], s=48, depthshade=False)
    current_cells_scatter = ax.scatter([], [], [], s=28, marker="s", depthshade=False)
    future_cells_scatter = ax.scatter([], [], [], s=16, marker="s", alpha=0.35, depthshade=False)
    title_text = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    label_artists: list = []
    max_players_per_frame = int(df.groupby("tick")["steamid"].nunique().max())
    for _ in range(max_players_per_frame):
        label_artists.append(ax.text(0.0, 0.0, 0.0, "", fontsize=7, visible=False))

    def update(frame_index: int):
        tick_value, tick_rows = frames[frame_index]
        positions_xyz = tick_rows[[x_col, y_col, z_col]].to_numpy(dtype=float)
        current_cells_xyz = tick_rows[["cell_center_x", "cell_center_y", "cell_center_z"]].to_numpy(dtype=float)
        colors = [team_colors.get(int(team_num), unknown_color) for team_num in tick_rows["team_num"].tolist()]

        set_scatter_xyz(players_scatter, positions_xyz)
        players_scatter.set_color(colors)
        set_scatter_xyz(current_cells_scatter, current_cells_xyz)
        current_cells_scatter.set_color(colors)

        future_xyz_rows: list[np.ndarray] = []
        future_colors: list[str] = []
        for row in tick_rows.itertuples(index=False):
            track = player_tracks.get(int(getattr(row, "steamid")))
            if track is None:
                continue
            future_xyz = track.future_cells_xyz(int(getattr(row, "player_track_index")), int(args.future_cells))
            if future_xyz.size == 0:
                continue
            base_color = team_colors.get(int(getattr(row, "team_num")), unknown_color)
            future_xyz_rows.append(future_xyz)
            future_colors.extend([base_color] * len(future_xyz))
        all_future_xyz = np.concatenate(future_xyz_rows, axis=0) if future_xyz_rows else np.empty((0, 3), dtype=float)
        set_scatter_xyz(future_cells_scatter, all_future_xyz)
        future_cells_scatter.set_color(future_colors if future_colors else [])

        label_cursor = 0
        if args.show_labels:
            for row in tick_rows.itertuples(index=False):
                artist = label_artists[label_cursor]
                artist.set_position((float(getattr(row, x_col)), float(getattr(row, y_col))))
                artist.set_3d_properties(float(getattr(row, z_col)), 'z')
                artist.set_text(str(getattr(row, "name")))
                artist.set_visible(True)
                label_cursor += 1
                
        for i in range(label_cursor, len(label_artists)):
            if not label_artists[i].get_visible():
                break
            label_artists[i].set_visible(False)

        alive_count = int(tick_rows["is_alive"].fillna(False).astype(bool).sum()) if "is_alive" in tick_rows.columns else len(tick_rows)
        title_text.set_text(
            f"tick={tick_value} players={len(tick_rows)} alive={alive_count} future_cells={len(all_future_xyz)}"
        )
        return [players_scatter, current_cells_scatter, future_cells_scatter, title_text, *label_artists]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=True,
        repeat=True,
    )
    update(0)

    print(f"Loaded: {selected_path}")
    print(f"Rows: {len(df)} | Unique ticks: {df['tick'].nunique()} | Rendered frames: {len(frames)}")
    print(
        f"Players: {df['steamid'].nunique()} | XYZ columns: {x_col}/{y_col}/{z_col} | "
        f"Future cells per player: {args.future_cells}"
    )
    print(
        f"Playback: tick_step={max(1, int(args.tick_step))} "
        f"ticks_per_second={float(args.ticks_per_second):.2f} interval_ms={interval_ms}"
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
