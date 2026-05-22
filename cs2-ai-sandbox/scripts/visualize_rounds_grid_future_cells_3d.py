from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

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
            return np.empty((0, 3), dtype=np.float32)
        segment_index = int(self.row_to_segment[int(row_track_index)])
        future_start = segment_index + 1
        future_end = min(future_start + int(future_cells), len(self.segment_centers_xyz))
        if future_start >= future_end:
            return np.empty((0, 3), dtype=np.float32)
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
    parser.add_argument("--ticks-per-second", type=float, default=40, help="Playback speed in rendered ticks per second.")
    parser.add_argument("--future-cells", type=int, default=20, help="How many future grid cells to highlight per player.")
    parser.add_argument("--show-labels", action="store_true", help="Show player name labels next to points.")
    parser.add_argument("--elev", type=float, default=35.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-60.0, help="3D camera azimuth.")
    parser.add_argument("--print-fps", action="store_true", help="Print measured FPS every second.")
    parser.add_argument("--grid-size", type=float, default=100.0, help="Scale factor for the 3D grid.")
    parser.add_argument("--grid-offset-x", type=float, default=0.0, help="X translation for the 3D grid.")
    parser.add_argument("--grid-offset-y", type=float, default=0.0, help="Y translation for the 3D grid.")
    parser.add_argument("--grid-offset-z", type=float, default=0.0, help="Z translation for the 3D grid.")
    parser.add_argument("--cross-size", type=float, default=25.0, help="Size of the cross for dead players.")
    parser.add_argument("--bg-image", type=Path, default=None, help="Path to a top-down radar image (e.g., .png).")
    parser.add_argument("--bg-scale", type=float, default=1.0, help="Scale factor for the background image.")
    return parser.parse_args()


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

@dataclass(slots=True)
class FrameRenderData:
    tick_value: int
    alive_positions_xyz: np.ndarray
    alive_colors_rgba: np.ndarray
    dead_lines_xyz: np.ndarray
    dead_line_colors_rgba: np.ndarray
    current_cells_xyz: np.ndarray
    future_cells_xyz: np.ndarray
    current_cell_colors_rgba: np.ndarray
    future_cell_colors_rgba: np.ndarray
    names: list[str]
    alive_count: int

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    hex_color = hex_color.lstrip('#')
    return [
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
        alpha
    ]

TEAM_COLORS = {
    2: hex_to_rgba("#d94f30", 1.0),
    3: hex_to_rgba("#3b82f6", 1.0),
}
UNKNOWN_COLOR = hex_to_rgba("#9ca3af", 1.0)

def precompute_frames(frames: list[tuple[int, pd.DataFrame]], player_tracks: dict[int, PlayerTrack3D], args: argparse.Namespace, x_col: str, y_col: str, z_col: str) -> list[FrameRenderData]:
    print("Pre-caching frames...")
    cached_frames = []
    for tick_value, tick_rows in frames:
        positions_xyz = tick_rows[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
        current_cells_xyz = tick_rows[["cell_center_x", "cell_center_y", "cell_center_z"]].to_numpy(dtype=np.float32)
        
        colors_list = []
        for team_num in tick_rows["team_num"].tolist():
            colors_list.append(TEAM_COLORS.get(int(team_num), UNKNOWN_COLOR))
        all_colors_rgba = np.array(colors_list, dtype=np.float32)
        current_cell_colors_rgba = all_colors_rgba.copy()

        alive_mask = tick_rows["is_alive"].fillna(False).to_numpy(dtype=bool) if "is_alive" in tick_rows.columns else np.ones(len(tick_rows), dtype=bool)
        
        alive_positions_xyz = positions_xyz[alive_mask]
        alive_colors_rgba = all_colors_rgba[alive_mask]
        
        dead_positions = positions_xyz[~alive_mask]
        dead_colors = all_colors_rgba[~alive_mask]
        
        D = len(dead_positions)
        dead_lines_xyz = np.empty((D * 4, 3), dtype=np.float32)
        dead_line_colors_rgba = np.empty((D * 4, 4), dtype=np.float32)
        
        S = float(args.cross_size)
        if D > 0:
            for i in range(D):
                x, y, z = dead_positions[i]
                dead_lines_xyz[i*4 + 0] = [x - S, y - S, z]
                dead_lines_xyz[i*4 + 1] = [x + S, y + S, z]
                dead_lines_xyz[i*4 + 2] = [x - S, y + S, z]
                dead_lines_xyz[i*4 + 3] = [x + S, y - S, z]
                dead_line_colors_rgba[i*4:i*4+4] = dead_colors[i]

        names = tick_rows["name"].tolist() if "name" in tick_rows.columns else []
        alive_count = int(alive_mask.sum())

        future_xyz_list = []
        future_colors_list = []
        
        if args.future_cells > 0:
            for row in tick_rows.itertuples(index=False):
                steamid = int(getattr(row, "steamid"))
                track = player_tracks.get(steamid)
                if track is None:
                    continue
                future_xyz = track.future_cells_xyz(int(getattr(row, "player_track_index")), int(args.future_cells))
                if future_xyz.size > 0:
                    future_xyz_list.append(future_xyz)
                    base_color = TEAM_COLORS.get(int(getattr(row, "team_num")), UNKNOWN_COLOR)
                    future_color = [base_color[0], base_color[1], base_color[2], 0.35]
                    future_colors_list.extend([future_color] * len(future_xyz))
                
        if future_xyz_list:
            future_cells_xyz = np.concatenate(future_xyz_list, axis=0).astype(np.float32)
            future_cell_colors_rgba = np.array(future_colors_list, dtype=np.float32)
        else:
            future_cells_xyz = np.empty((0, 3), dtype=np.float32)
            future_cell_colors_rgba = np.empty((0, 4), dtype=np.float32)

        cached_frames.append(FrameRenderData(
            tick_value=tick_value,
            alive_positions_xyz=alive_positions_xyz,
            alive_colors_rgba=alive_colors_rgba,
            dead_lines_xyz=dead_lines_xyz,
            dead_line_colors_rgba=dead_line_colors_rgba,
            current_cells_xyz=current_cells_xyz,
            future_cells_xyz=future_cells_xyz,
            current_cell_colors_rgba=current_cell_colors_rgba,
            future_cell_colors_rgba=future_cell_colors_rgba,
            names=names,
            alive_count=alive_count
        ))
    return cached_frames


class RoundRealtimeViewer(QtWidgets.QMainWindow):
    def __init__(self, cached_frames: list[FrameRenderData], args: argparse.Namespace):
        super().__init__()
        self.cached_frames = cached_frames
        self.args = args
        self.current_frame_index = 0
        self.is_playing = True
        
        self.setWindowTitle("Realtime 3D Round Viewer")
        self.resize(1024, 768)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Top panel
        top_panel = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Status: Initializing...")
        top_panel.addWidget(self.status_label)
        
        self.play_pause_btn = QtWidgets.QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        top_panel.addWidget(self.play_pause_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_playback)
        top_panel.addWidget(self.reset_btn)

        top_panel.addWidget(QtWidgets.QLabel("FPS:"))
        self.fps_spinbox = QtWidgets.QDoubleSpinBox()
        self.fps_spinbox.setRange(1.0, 240.0)
        self.fps_spinbox.setValue(float(self.args.ticks_per_second))
        self.fps_spinbox.valueChanged.connect(self.update_timer_interval)
        top_panel.addWidget(self.fps_spinbox)
        main_layout.addLayout(top_panel)

        # GL View
        self.gl_view = gl.GLViewWidget()
        main_layout.addWidget(self.gl_view, 1)

        self.grid = gl.GLGridItem()
        self.grid.scale(self.args.grid_size, self.args.grid_size, self.args.grid_size)
        self.grid.translate(self.args.grid_offset_x, self.args.grid_offset_y, self.args.grid_offset_z)
        self.gl_view.addItem(self.grid)

        if self.args.bg_image and self.args.bg_image.exists():
            import matplotlib.pyplot as plt
            # Load image (returns height, width, channels)
            img_data = plt.imread(self.args.bg_image)
            
            # Convert float [0,1] to uint8 [0,255] which GLImageItem expects
            if img_data.dtype in (np.float32, np.float64):
                img_data = np.clip(img_data * 255, 0, 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)
                
            # Make sure it's RGBA (4 channels)
            if img_data.shape[2] == 3:
                alpha = np.full((img_data.shape[0], img_data.shape[1], 1), 255, dtype=np.uint8)
                img_data = np.concatenate([img_data, alpha], axis=2)

            # Transpose from (H, W, C) to (W, H, C) for pyqtgraph
            img_data = np.transpose(img_data, (1, 0, 2))
            
            # Flip Y to match OpenGL coordinate system (bottom-left origin)
            img_data = np.flip(img_data, axis=1)

            img_item = gl.GLImageItem(img_data)
            
            # Center the image
            w, h = img_data.shape[0], img_data.shape[1]
            scale = self.args.bg_scale
            img_item.scale(scale, scale, 1.0)
            
            # Move the image slightly below Z=0 and center it
            img_item.translate(-w/2 * scale + self.args.grid_offset_x, -h/2 * scale + self.args.grid_offset_y, self.args.grid_offset_z - 2.0)
            self.gl_view.addItem(img_item)

        self.alive_scatter = gl.GLScatterPlotItem(size=8, pxMode=True)
        self.gl_view.addItem(self.alive_scatter)

        self.dead_lines = gl.GLLinePlotItem(mode='lines', width=2.0, antialias=True)
        self.gl_view.addItem(self.dead_lines)

        self.current_cells_scatter = gl.GLScatterPlotItem(size=6, pxMode=True)
        self.gl_view.addItem(self.current_cells_scatter)

        self.future_cells_scatter = gl.GLScatterPlotItem(size=4, pxMode=True)
        self.gl_view.addItem(self.future_cells_scatter)

        # Labels (Fallback to 2D overlay)
        self.labels_overlay = QtWidgets.QLabel(self.gl_view)
        self.labels_overlay.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.labels_overlay.setStyleSheet("color: white; font-weight: bold;")
        self.labels_overlay.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        
        # Timeline slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.cached_frames) - 1)
        self.slider.sliderMoved.connect(self.slider_moved)
        main_layout.addWidget(self.slider)

        # Setup Camera
        if self.cached_frames:
            all_pos_list = [f.alive_positions_xyz for f in self.cached_frames if f.alive_positions_xyz.size > 0]
            if not all_pos_list:
                all_pos_list = [np.zeros((1,3))]
            all_pos = np.concatenate(all_pos_list)
            if all_pos.size > 0:
                min_pos = all_pos.min(axis=0)
                max_pos = all_pos.max(axis=0)
                center = (min_pos + max_pos) / 2.0
                distance = np.linalg.norm(max_pos - min_pos) * 1.5
                if distance == 0:
                    distance = 1000
                self.gl_view.setCameraPosition(pos=QtGui.QVector3D(*center), distance=distance, elevation=self.args.elev, azimuth=self.args.azim)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.update_timer_interval()
        
        self.last_fps_time = time.time()
        self.frames_rendered = 0
        self.measured_fps = 0.0

        self.render_current_frame()
        self.timer.start()

    def update_timer_interval(self):
        fps = self.fps_spinbox.value()
        interval_ms = max(1, round(1000.0 / fps))
        self.timer.setInterval(interval_ms)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.setText("Pause" if self.is_playing else "Play")

    def reset_playback(self):
        self.current_frame_index = 0
        self.slider.setValue(0)
        self.render_current_frame()

    def slider_moved(self, value):
        self.current_frame_index = value
        self.render_current_frame()

    def update_frame(self):
        if self.is_playing:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.cached_frames)
            self.slider.setValue(self.current_frame_index)
            self.render_current_frame()
            
        self.frames_rendered += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.measured_fps = self.frames_rendered / (now - self.last_fps_time)
            if self.args.print_fps:
                print(f"Measured FPS: {self.measured_fps:.1f}")
            self.frames_rendered = 0
            self.last_fps_time = now

    def render_current_frame(self):
        if not self.cached_frames:
            return
            
        frame_data = self.cached_frames[self.current_frame_index]
        
        if frame_data.alive_positions_xyz.size > 0:
            self.alive_scatter.setData(pos=frame_data.alive_positions_xyz, color=frame_data.alive_colors_rgba, size=8, pxMode=True)
        else:
            self.alive_scatter.setData(pos=np.empty((0,3), dtype=np.float32), color=np.empty((0,4), dtype=np.float32))

        if frame_data.dead_lines_xyz.size > 0:
            self.dead_lines.setData(pos=frame_data.dead_lines_xyz, color=frame_data.dead_line_colors_rgba, width=2.0, mode='lines')
        else:
            self.dead_lines.setData(pos=np.empty((0,3), dtype=np.float32), color=np.empty((0,4), dtype=np.float32))

        if frame_data.current_cells_xyz.size > 0:
            self.current_cells_scatter.setData(pos=frame_data.current_cells_xyz, color=frame_data.current_cell_colors_rgba, size=6, pxMode=True)
        else:
            self.current_cells_scatter.setData(pos=np.empty((0,3), dtype=np.float32), color=np.empty((0,4), dtype=np.float32))

        if frame_data.future_cells_xyz.size > 0:
            self.future_cells_scatter.setData(pos=frame_data.future_cells_xyz, color=frame_data.future_cell_colors_rgba, size=4, pxMode=True)
        else:
            self.future_cells_scatter.setData(pos=np.empty((0,3), dtype=np.float32), color=np.empty((0,4), dtype=np.float32))
            
        if self.args.show_labels:
            text = f"Tick: {frame_data.tick_value}\nPlayers:\n" + "\n".join(frame_data.names)
            self.labels_overlay.setText(text)
            self.labels_overlay.adjustSize()
        else:
            self.labels_overlay.setText("")

        status_text = (f"Tick: {frame_data.tick_value} | Frame: {self.current_frame_index + 1}/{len(self.cached_frames)} | "
                       f"Players: {len(frame_data.alive_positions_xyz) + len(frame_data.dead_lines_xyz)//4} | Alive: {frame_data.alive_count} | "
                       f"Future Cells: {len(frame_data.future_cells_xyz)} | FPS: {self.measured_fps:.1f}")
        self.status_label.setText(status_text)


def main() -> int:
    args = parse_args()
    selected_path = select_round_file(args)
    if not selected_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {selected_path}")

    df, (x_col, y_col, z_col) = load_round_dataframe(selected_path)
    frames = build_tick_frames(df, tick_step=max(1, int(args.tick_step)), max_ticks=args.max_ticks)
    if not frames:
        raise ValueError("No frames to render after filtering.")

    player_tracks = build_player_tracks(df)
    
    cached_frames = precompute_frames(frames, player_tracks, args, x_col, y_col, z_col)

    interval_ms = max(1, round(1000.0 / float(args.ticks_per_second)))
    
    print(f"Loaded: {selected_path}")
    print(f"Rows: {len(df)} | Unique ticks: {df['tick'].nunique()} | Rendered frames: {len(frames)}")
    print(
        f"Players: {df['steamid'].nunique()} | XYZ columns: {x_col}/{y_col}/{z_col} | "
        f"Future cells per player: {args.future_cells}"
    )
    print(f"Playback: Target FPS={float(args.ticks_per_second):.2f} Timer interval={interval_ms}ms")

    app = QtWidgets.QApplication(sys.argv)
    viewer = RoundRealtimeViewer(cached_frames, args)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
