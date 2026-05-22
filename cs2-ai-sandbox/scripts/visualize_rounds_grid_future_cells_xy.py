from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs2_ai.navigation.cell_indexer import build_grid_map
from cs2_ai.navigation.grid_map import GridMap


POSITION_COLUMN_CANDIDATES = (
    ("X", "Y"),
    ("x", "y"),
    ("pos_x", "pos_y"),
    ("self_x", "self_y"),
    ("player_x", "player_y"),
)


@dataclass(slots=True)
class PlayerTrack:
    steamid: int
    name: str
    team_num: int
    ticks: np.ndarray
    row_track_indices: np.ndarray
    row_to_segment: np.ndarray
    segment_start_indices: np.ndarray
    segment_centers_xy: np.ndarray

    def future_cells_xy(self, row_track_index: int, future_cells: int) -> np.ndarray:
        if future_cells <= 0:
            return np.empty((0, 2), dtype=float)
        segment_index = int(self.row_to_segment[int(row_track_index)])
        future_start = segment_index + 1
        future_end = min(future_start + int(future_cells), len(self.segment_start_indices))
        if future_start >= future_end:
            return np.empty((0, 2), dtype=float)
        return self.segment_centers_xy[future_start:future_end]


def resolve_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    columns = set(df.columns)
    for x_col, y_col in POSITION_COLUMN_CANDIDATES:
        if x_col in columns and y_col in columns:
            return x_col, y_col
    raise ValueError(
        "Could not resolve X/Y columns. Expected one of: "
        "X/Y, x/y, pos_x/pos_y, self_x/self_y, player_x/player_y."
    )


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate one grid-labeled round parquet file and overlay future grid cells per player."
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
    parser.add_argument("--tick-step", type=int, default=1, help="Render every Nth tick.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Optional limit on rendered ticks.")
    parser.add_argument("--interval-ms", type=int, default=40, help="Animation interval in milliseconds.")
    parser.add_argument(
        "--ticks-per-second",
        type=float,
        default=25.0,
        help="Playback speed in rendered ticks per second. Used to derive animation interval.",
    )
    parser.add_argument("--show-labels", action="store_true", help="Show player name labels next to points.")
    parser.add_argument("--future-cells", type=int, default=30, help="How many future grid cells to highlight per player.")
    parser.add_argument("--map", type=str, default="de_dust2", help="Grid map name.")
    parser.add_argument(
        "--annotate-future-order",
        action="store_true",
        help="Draw 1..N order labels on future cells.",
    )
    return parser.parse_args()


def resolve_interval_ms(args: argparse.Namespace) -> int:
    ticks_per_second = float(args.ticks_per_second)
    if ticks_per_second <= 0:
        raise ValueError(f"--ticks-per-second must be > 0, got {ticks_per_second}")
    tick_step = max(1, int(args.tick_step))
    computed_interval_ms = int(round((1000.0 * tick_step) / ticks_per_second))
    return max(1, computed_interval_ms if computed_interval_ms > 0 else int(args.interval_ms))


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


def select_round_file(args: argparse.Namespace) -> Path:
    if args.input is not None:
        return args.input
    candidates = sorted(args.input_dir.glob(args.input_glob))
    if not candidates:
        raise FileNotFoundError(f"No round parquet files found for {args.input_dir} / {args.input_glob}")
    rng = random.Random(args.seed)
    return rng.choice(candidates)


def build_player_tracks(df: pd.DataFrame) -> dict[int, PlayerTrack]:
    tracks: dict[int, PlayerTrack] = {}
    for steamid, group in df.groupby("steamid", sort=False, dropna=False):
        ordered = group.sort_values(["tick", "player_track_index"]).reset_index(drop=True)
        cell_ids = pd.to_numeric(ordered["current_cell_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        centers_xy = ordered[["cell_center_x", "cell_center_y"]].to_numpy(dtype=float, copy=False)
        segment_start_mask = np.ones(len(ordered), dtype=bool)
        if len(ordered) > 1:
            segment_start_mask[1:] = cell_ids[1:] != cell_ids[:-1]
        segment_start_indices = np.flatnonzero(segment_start_mask).astype(np.int64)
        row_to_segment = np.empty(len(ordered), dtype=np.int64)
        next_starts = np.append(segment_start_indices[1:], len(ordered))
        for segment_idx, (start_idx, end_idx) in enumerate(zip(segment_start_indices, next_starts, strict=True)):
            row_to_segment[start_idx:end_idx] = segment_idx
        tracks[int(steamid)] = PlayerTrack(
            steamid=int(steamid),
            name=str(ordered["name"].iloc[0]),
            team_num=int(pd.to_numeric(ordered["team_num"], errors="coerce").fillna(0).iloc[0]),
            ticks=pd.to_numeric(ordered["tick"], errors="coerce").to_numpy(dtype=np.int64, copy=False),
            row_track_indices=pd.to_numeric(ordered["player_track_index"], errors="coerce").to_numpy(dtype=np.int64, copy=False),
            row_to_segment=row_to_segment,
            segment_start_indices=segment_start_indices,
            segment_centers_xy=centers_xy[segment_start_indices],
        )
    return tracks


def load_round_dataframe(path: Path, map_name_fallback: str) -> tuple[pd.DataFrame, str, str, GridMap]:
    df = pd.read_parquet(path)
    required_columns = [
        "tick",
        "steamid",
        "name",
        "team_num",
        "current_cell_id",
        "cell_center_x",
        "cell_center_y",
    ]
    require_columns(df, required_columns)
    x_col, y_col = resolve_xy_columns(df)
    df = df.dropna(subset=[x_col, y_col, "tick", "steamid", "current_cell_id", "cell_center_x", "cell_center_y"]).copy()
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").astype("int64")
    df["steamid"] = pd.to_numeric(df["steamid"], errors="coerce").astype("int64")
    df["team_num"] = pd.to_numeric(df["team_num"], errors="coerce").fillna(0).astype("int64")
    df["current_cell_id"] = pd.to_numeric(df["current_cell_id"], errors="coerce").astype("int64")
    df["cell_center_x"] = pd.to_numeric(df["cell_center_x"], errors="coerce")
    df["cell_center_y"] = pd.to_numeric(df["cell_center_y"], errors="coerce")
    df = df.sort_values(["steamid", "tick"]).reset_index(drop=True)
    df["player_track_index"] = df.groupby("steamid", sort=False).cumcount().astype("int64")
    map_name = str(df["map"].dropna().iloc[0]) if "map" in df.columns and not df["map"].dropna().empty else str(map_name_fallback)
    return df, x_col, y_col, build_grid_map(map_name)


def main() -> int:
    args = parse_args()
    interval_ms = resolve_interval_ms(args)
    selected_path = select_round_file(args)
    if not selected_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {selected_path}")

    df, x_col, y_col, grid_map = load_round_dataframe(selected_path, args.map)
    frames = build_tick_frames(df, tick_step=max(1, int(args.tick_step)), max_ticks=args.max_ticks)
    if not frames:
        raise ValueError("No frames to render after filtering.")

    player_tracks = build_player_tracks(df)
    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    y_min = float(df[y_col].min())
    y_max = float(df[y_col].max())
    x_margin = max((x_max - x_min) * 0.05, 50.0)
    y_margin = max((y_max - y_min) * 0.05, 50.0)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_title(selected_path.name)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    team_colors = {
        2: "#d94f30",
        3: "#3b82f6",
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
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    label_artists: list[plt.Text] = []
    max_players_per_frame = int(df.groupby("tick")["steamid"].nunique().max())
    for _ in range(max_players_per_frame):
        label_artists.append(
            ax.text(0.0, 0.0, "", fontsize=8, ha="left", va="bottom", visible=False)
        )
    max_patch_count = max_players_per_frame * (1 + max(0, int(args.future_cells)))
    future_patches: list[Rectangle] = []
    for _ in range(max_patch_count):
        patch = Rectangle((0.0, 0.0), 0.0, 0.0, visible=False)
        ax.add_patch(patch)
        future_patches.append(patch)
    max_order_text_count = max_players_per_frame * max(0, int(args.future_cells))
    future_order_artists: list[plt.Text] = []
    for _ in range(max_order_text_count):
        artist = ax.text(0.0, 0.0, "", fontsize=6, ha="center", va="center", color="white", visible=False)
        future_order_artists.append(artist)

    def update(frame_index: int):
        tick_value, tick_rows = frames[frame_index]
        offsets = tick_rows[[x_col, y_col]].to_numpy(dtype=float)
        colors = [team_colors.get(int(team_num), unknown_color) for team_num in tick_rows["team_num"].tolist()]
        scatter.set_offsets(offsets)
        scatter.set_color(colors)

        rendered_cells = 0
        patch_cursor = 0
        order_cursor = 0
        label_cursor = 0

        if args.show_labels:
            for row in tick_rows.itertuples(index=False):
                artist = label_artists[label_cursor]
                artist.set_position((float(getattr(row, x_col)), float(getattr(row, y_col))))
                artist.set_text(str(getattr(row, "name")))
                artist.set_visible(True)
                label_cursor += 1

        for row in tick_rows.itertuples(index=False):
            steamid = int(getattr(row, "steamid"))
            track = player_tracks.get(steamid)
            if track is None:
                continue
            base_color = team_colors.get(int(getattr(row, "team_num")), unknown_color)
            current_center_x = float(getattr(row, "cell_center_x"))
            current_center_y = float(getattr(row, "cell_center_y"))
            current_patch = future_patches[patch_cursor]
            patch_cursor += 1
            current_patch.set_xy(
                (
                    current_center_x - grid_map.config.cell_size_xy / 2.0,
                    current_center_y - grid_map.config.cell_size_xy / 2.0,
                )
            )
            current_patch.set_width(grid_map.config.cell_size_xy)
            current_patch.set_height(grid_map.config.cell_size_xy)
            current_patch.set_facecolor("none")
            current_patch.set_edgecolor(base_color)
            current_patch.set_linewidth(1.8)
            current_patch.set_alpha(0.9)
            current_patch.set_visible(True)

            future_cells_xy = track.future_cells_xy(int(getattr(row, "player_track_index")), int(args.future_cells))
            for order_index, (future_x, future_y) in enumerate(future_cells_xy, start=1):
                alpha = max(0.08, 0.42 * (1.0 - ((order_index - 1) / max(1, int(args.future_cells)))))
                patch = future_patches[patch_cursor]
                patch_cursor += 1
                patch.set_xy(
                    (
                        float(future_x) - grid_map.config.cell_size_xy / 2.0,
                        float(future_y) - grid_map.config.cell_size_xy / 2.0,
                    )
                )
                patch.set_width(grid_map.config.cell_size_xy)
                patch.set_height(grid_map.config.cell_size_xy)
                patch.set_facecolor(base_color)
                patch.set_edgecolor(base_color)
                patch.set_linewidth(0.8)
                patch.set_alpha(alpha)
                patch.set_visible(True)
                rendered_cells += 1
                if args.annotate_future_order:
                    order_artist = future_order_artists[order_cursor]
                    order_cursor += 1
                    order_artist.set_position((float(future_x), float(future_y)))
                    order_artist.set_text(str(order_index))
                    order_artist.set_visible(True)

        for i in range(label_cursor, len(label_artists)):
            if not label_artists[i].get_visible():
                break
            label_artists[i].set_visible(False)
        for i in range(patch_cursor, len(future_patches)):
            if not future_patches[i].get_visible():
                break
            future_patches[i].set_visible(False)
        for i in range(order_cursor, len(future_order_artists)):
            if not future_order_artists[i].get_visible():
                break
            future_order_artists[i].set_visible(False)

        alive_count = int(tick_rows["is_alive"].fillna(False).astype(bool).sum()) if "is_alive" in tick_rows.columns else len(tick_rows)
        title_text.set_text(
            f"tick={tick_value} players={len(tick_rows)} alive={alive_count} future_cells={rendered_cells}"
        )
        return [scatter, title_text, *label_artists, *future_patches, *future_order_artists]

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
        f"Players: {df['steamid'].nunique()} | X/Y columns: {x_col}/{y_col} | "
        f"Future cells per player: {args.future_cells}"
    )
    print(
        f"Grid: map={grid_map.config.map_name} cell_size_xy={grid_map.config.cell_size_xy} "
        f"cell_size_z={grid_map.config.cell_size_z}"
    )
    print(
        f"Playback: tick_step={max(1, int(args.tick_step))} "
        f"ticks_per_second={float(args.ticks_per_second):.2f} interval_ms={interval_ms}"
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
