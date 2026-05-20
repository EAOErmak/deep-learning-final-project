from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from demoparser2 import DemoParser

TICK_FIELDS = [
    "X",
    "Y",
    "Z",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "health",
    "armor_value",
    "has_helmet",
    "is_alive",
    "team_num",
    "balance",
    "start_balance",
    "cash_spent_this_round",
    "pitch",
    "yaw",
    "active_weapon_name",
    "active_weapon_ammo",
    "total_ammo_left",
    "is_scoped",
    "is_walking",
    "is_airborne",
    "duck_amount",
    "ducking",
    "in_crouch",
    "shots_fired",
    "flash_duration",
    "spotted",
    "approximate_spotted_by",
    "last_place_name",
    "in_bomb_zone",
    "in_buy_zone",
    "which_bomb_zone",
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "FIRE",
    "RIGHTCLICK",
    "RELOAD",
    "USE",
    "ZOOM",
    "WALK",
    "buttons",
    "usercmd_viewangle_x",
    "usercmd_viewangle_y",
    "usercmd_mouse_dx",
    "usercmd_mouse_dy",
    "usercmd_forward_move",
    "usercmd_left_move",
    "usercmd_buttonstate_1",
    "usercmd_buttonstate_2",
    "usercmd_weapon_select",
    "is_freeze_period",
    "is_warmup_period",
    "round_in_progress",
    "round_start_time",
    "total_rounds_played",
    "game_phase",
    "is_bomb_dropped",
    "is_bomb_planted",
    "round_win_status",
    "round_win_reason",
    "ct_losing_streak",
    "t_losing_streak",
]

EVENTS = [
    "round_start",
    "round_end",
    "player_death",
    "player_hurt",
    "weapon_fire",
    "bomb_planted",
    "bomb_defused",
    "bomb_dropped",
    "bomb_pickup",
]

NUMERIC_FILL_ZERO = [
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "active_weapon_ammo",
    "total_ammo_left",
    "usercmd_mouse_dx",
    "usercmd_mouse_dy",
    "usercmd_forward_move",
    "usercmd_left_move",
    "usercmd_viewangle_x",
    "usercmd_viewangle_y",
    "usercmd_buttonstate_1",
    "usercmd_buttonstate_2",
    "duck_amount",
    "flash_duration",
    "shots_fired",
    "armor_value",
    "balance",
    "start_balance",
    "cash_spent_this_round",
]

STRING_FILL = {
    "active_weapon_name": "none",
    "last_place_name": "unknown",
    "name": "unknown",
}

BOOL_FILL_FALSE = [
    "has_helmet",
    "is_alive",
    "is_scoped",
    "is_walking",
    "is_airborne",
    "ducking",
    "spotted",
    "in_bomb_zone",
    "in_buy_zone",
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "FIRE",
    "RIGHTCLICK",
    "RELOAD",
    "USE",
    "ZOOM",
    "WALK",
    "is_bomb_planted",
    "is_bomb_dropped",
    "is_freeze_period",
    "is_warmup_period",
    "round_in_progress",
]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    demos_dir: Path
    dataset_dir: Path
    registry_file: Path
    raw_ticks_dir: Path
    clean_play_ticks_dir: Path
    clean_buy_ticks_dir: Path
    round_events_dir: Path
    events_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and clean one CS2 demo into dataset layers.")
    parser.add_argument("--demo", type=str, default=None, help="Parse a specific .dem file.")
    parser.add_argument("--force", action="store_true", help="Reparse even if the demo is already completed.")
    parser.add_argument("--demos-dir", type=str, default="demos", help="Directory containing .dem files.")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory for output parquet files.")
    return parser.parse_args()


def build_paths(demos_dir_arg: str, dataset_dir_arg: str) -> Paths:
    project_root = Path(__file__).resolve().parent.parent
    demos_dir = (project_root / demos_dir_arg).resolve()
    dataset_dir = (project_root / dataset_dir_arg).resolve()
    return Paths(
        project_root=project_root,
        demos_dir=demos_dir,
        dataset_dir=dataset_dir,
        registry_file=dataset_dir / "parsed_demos.json",
        raw_ticks_dir=dataset_dir / "raw_ticks",
        clean_play_ticks_dir=dataset_dir / "clean_play_ticks",
        clean_buy_ticks_dir=dataset_dir / "clean_buy_ticks",
        round_events_dir=dataset_dir / "round_events",
        events_dir=dataset_dir / "events",
    )


def ensure_directories(paths: Paths) -> None:
    paths.demos_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_ticks_dir.mkdir(parents=True, exist_ok=True)
    paths.clean_play_ticks_dir.mkdir(parents=True, exist_ok=True)
    paths.clean_buy_ticks_dir.mkdir(parents=True, exist_ok=True)
    paths.round_events_dir.mkdir(parents=True, exist_ok=True)
    paths.events_dir.mkdir(parents=True, exist_ok=True)

    if not paths.registry_file.exists():
        paths.registry_file.write_text(json.dumps({"parsed": []}, indent=2), encoding="utf-8")


def load_registry(registry_file: Path) -> dict[str, Any]:
    if not registry_file.exists():
        return {"parsed": []}

    data = json.loads(registry_file.read_text(encoding="utf-8"))
    if not isinstance(data.get("parsed"), list):
        raise RuntimeError(f"Registry format is invalid: {registry_file}")
    return data


def save_registry(registry_file: Path, registry: dict[str, Any]) -> None:
    registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def make_registry_relpath(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def get_expected_output_paths(paths: Paths, demo_name: str) -> dict[str, Path]:
    demo_stem = Path(demo_name).stem
    return {
        "raw_ticks": paths.raw_ticks_dir / f"{demo_stem}_ticks.parquet",
        "clean_play_ticks": paths.clean_play_ticks_dir / f"{demo_stem}_play_ticks.parquet",
        "clean_buy_ticks": paths.clean_buy_ticks_dir / f"{demo_stem}_buy_ticks.parquet",
        "round_events": paths.round_events_dir / f"{demo_stem}_round_events.parquet",
    }


def registry_entries_by_demo(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in registry.get("parsed", []):
        if isinstance(entry, dict) and isinstance(entry.get("demo_name"), str):
            result[entry["demo_name"]] = entry
    return result


def is_registry_entry_complete(entry: dict[str, Any] | None, paths: Paths) -> bool:
    if not entry or entry.get("status") != "completed":
        return False

    files = entry.get("files")
    if not isinstance(files, dict):
        return False

    for key in ("raw_ticks", "clean_play_ticks", "clean_buy_ticks", "round_events"):
        rel_path = files.get(key)
        if not isinstance(rel_path, str):
            return False
        target = (paths.project_root / rel_path).resolve()
        if not target.exists():
            return False

    return True


def find_demo_to_process(
    paths: Paths,
    registry: dict[str, Any],
    force: bool,
    demo_arg: str | None,
) -> tuple[list[Path], Path | None]:
    demo_files = sorted(paths.demos_dir.glob("*.dem"))
    entries = registry_entries_by_demo(registry)

    if demo_arg:
        requested = Path(demo_arg)
        selected = requested if requested.is_absolute() else (paths.project_root / requested)
        selected = selected.resolve()
        if not selected.exists():
            raise FileNotFoundError(f"Demo file not found: {selected}")
        return demo_files, selected

    for demo_file in demo_files:
        entry = entries.get(demo_file.name)
        if force or not is_registry_entry_complete(entry, paths):
            return demo_files, demo_file

    return demo_files, None


def parse_single_event(parser: DemoParser, event_name: str) -> pd.DataFrame:
    if hasattr(parser, "parse_event"):
        event_df = parser.parse_event(event_name)
    elif hasattr(parser, "parse_events"):
        event_df = parser.parse_events(event_name)
    else:
        raise AttributeError("demoparser2 parser has no parse_event/parse_events method")

    return pd.DataFrame(event_df)


def normalize_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_FILL_ZERO:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def normalize_string_frame(df: pd.DataFrame) -> pd.DataFrame:
    for column, fill_value in STRING_FILL.items():
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)
    return df


def normalize_bool_frame(df: pd.DataFrame) -> pd.DataFrame:
    for column in BOOL_FILL_FALSE:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)
    return df


def finalize_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_numeric_frame(df)
    df = normalize_string_frame(df)
    df = normalize_bool_frame(df)

    sort_columns = [column for column in ("total_rounds_played", "tick", "steamid") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def _event_round_offset(round_lookup: pd.DataFrame, event_df: pd.DataFrame) -> int:
    if event_df.empty or "round" not in event_df.columns or round_lookup.empty:
        return 0
    event_rounds = pd.to_numeric(event_df["round"], errors="coerce").dropna()
    raw_rounds = pd.to_numeric(round_lookup["round_number"], errors="coerce").dropna()
    if event_rounds.empty or raw_rounds.empty:
        return 0
    return int(raw_rounds.min()) - int(event_rounds.min())


def _build_event_tick_windows(raw_df: pd.DataFrame, parsed_events: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    round_lookup = build_round_lookup(raw_df)
    if round_lookup.empty:
        return None

    round_start_df = parsed_events.get("round_start", pd.DataFrame())
    round_end_df = parsed_events.get("round_end", pd.DataFrame())
    if round_start_df.empty or round_end_df.empty:
        return None
    if "round" not in round_start_df.columns or "tick" not in round_start_df.columns:
        return None
    if "round" not in round_end_df.columns or "tick" not in round_end_df.columns:
        return None

    round_offset = _event_round_offset(round_lookup, round_end_df)
    start_enriched = round_start_df.copy()
    end_enriched = round_end_df.copy()
    start_enriched["round_number"] = pd.to_numeric(start_enriched["round"], errors="coerce").astype("float64") + float(round_offset)
    end_enriched["round_number"] = pd.to_numeric(end_enriched["round"], errors="coerce").astype("float64") + float(round_offset)
    start_enriched["tick"] = pd.to_numeric(start_enriched["tick"], errors="coerce")
    end_enriched["tick"] = pd.to_numeric(end_enriched["tick"], errors="coerce")
    start_enriched = start_enriched.dropna(subset=["round_number", "tick"])
    end_enriched = end_enriched.dropna(subset=["round_number", "tick"])
    if start_enriched.empty or end_enriched.empty:
        return None

    # round_start may fire more than once per round; the last one is the closest proxy to live start.
    start_ticks = start_enriched.groupby("round_number", dropna=False)["tick"].max().reset_index(name="event_live_start_tick")
    end_ticks = end_enriched.groupby("round_number", dropna=False)["tick"].min().reset_index(name="event_round_end_tick")
    windows = round_lookup.merge(start_ticks, on="round_number", how="left").merge(end_ticks, on="round_number", how="left")
    return windows


def build_live_play_mask(raw_df: pd.DataFrame, parsed_events: dict[str, pd.DataFrame]) -> pd.Series:
    if not {"is_warmup_period", "is_freeze_period", "tick", "total_rounds_played"}.issubset(raw_df.columns):
        raise RuntimeError("Raw ticks are missing required play-filter columns.")

    tick_series = pd.to_numeric(raw_df["tick"], errors="coerce")
    round_series = pd.to_numeric(raw_df["total_rounds_played"], errors="coerce")
    base_mask = (
        (raw_df["is_warmup_period"] == False)
        & (raw_df["is_freeze_period"] == False)
        & tick_series.notna()
        & round_series.notna()
    )

    event_windows = _build_event_tick_windows(raw_df, parsed_events)
    if event_windows is None or event_windows.empty:
        if "round_in_progress" not in raw_df.columns:
            raise RuntimeError("Unable to derive live play mask: round events unavailable and round_in_progress missing.")
        return base_mask & (raw_df["round_in_progress"] == True)

    live_mask = pd.Series(False, index=raw_df.index)
    for row in event_windows.itertuples(index=False):
        round_mask = round_series == float(row.round_number)
        if pd.notna(row.event_live_start_tick):
            round_mask &= tick_series >= float(row.event_live_start_tick)
        if pd.notna(row.event_round_end_tick):
            round_mask &= tick_series < float(row.event_round_end_tick)
        elif pd.notna(row.end_tick):
            round_mask &= tick_series <= float(row.end_tick)
        live_mask |= round_mask

    if "round_in_progress" in raw_df.columns:
        live_confirmed = base_mask & live_mask & (raw_df["round_in_progress"] == True)
        if bool(live_confirmed.any()):
            return live_confirmed

    return base_mask & live_mask


def create_clean_play_ticks(raw_df: pd.DataFrame, parsed_events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df_play = raw_df[build_live_play_mask(raw_df, parsed_events)].copy()
    return finalize_clean_frame(df_play)


def create_clean_buy_ticks(raw_df: pd.DataFrame) -> pd.DataFrame:
    if not {"is_warmup_period", "is_freeze_period"}.issubset(raw_df.columns):
        raise RuntimeError("Raw ticks are missing required buy-filter columns.")

    df_buy = raw_df[
        (raw_df["is_warmup_period"] == False)
        & (raw_df["is_freeze_period"] == True)
    ].copy()
    return finalize_clean_frame(df_buy)


def build_round_lookup(raw_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"total_rounds_played", "tick"}
    if not required_columns.issubset(raw_df.columns):
        raise RuntimeError("Raw ticks are missing total_rounds_played/tick columns.")

    round_lookup = (
        raw_df.groupby("total_rounds_played", dropna=False)["tick"]
        .agg(start_tick="min", end_tick="max")
        .reset_index()
    )
    return round_lookup.rename(columns={"total_rounds_played": "round_number"})


def add_round_number_from_tick(event_df: pd.DataFrame, round_lookup: pd.DataFrame) -> pd.DataFrame:
    df = event_df.copy()
    if "total_rounds_played" in df.columns:
        df["round_number"] = df["total_rounds_played"]
        return df

    if "round_num" in df.columns:
        df["round_number"] = df["round_num"]
        return df

    if "tick" not in df.columns:
        df["round_number"] = pd.NA
        return df

    df["round_number"] = pd.NA
    for row in round_lookup.itertuples(index=False):
        mask = (df["tick"] >= row.start_tick) & (df["tick"] <= row.end_tick)
        df.loc[mask, "round_number"] = row.round_number

    return df


def get_last_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[-1]


def summarize_event_ticks(event_df: pd.DataFrame, round_lookup: pd.DataFrame, event_name: str) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["round_number", f"{event_name}_tick"])

    enriched = add_round_number_from_tick(event_df, round_lookup)
    enriched = enriched.dropna(subset=["round_number"])
    if enriched.empty or "tick" not in enriched.columns:
        return pd.DataFrame(columns=["round_number", f"{event_name}_tick"])

    grouped = enriched.groupby("round_number", dropna=False)["tick"].min().reset_index()
    return grouped.rename(columns={"tick": f"{event_name}_tick"})


def summarize_event_counts(event_df: pd.DataFrame, round_lookup: pd.DataFrame, output_column: str) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["round_number", output_column])

    enriched = add_round_number_from_tick(event_df, round_lookup)
    enriched = enriched.dropna(subset=["round_number"])
    if enriched.empty:
        return pd.DataFrame(columns=["round_number", output_column])

    return enriched.groupby("round_number", dropna=False).size().reset_index(name=output_column)


def infer_winner_columns(round_events_df: pd.DataFrame, parsed_events: dict[str, pd.DataFrame], round_lookup: pd.DataFrame) -> pd.DataFrame:
    df = round_events_df.copy()
    df["winner_team_num"] = pd.NA
    df["loser_team_num"] = pd.NA
    df["winner_team_num_known"] = False

    round_end_df = parsed_events.get("round_end", pd.DataFrame())
    if round_end_df.empty or "winner" not in round_end_df.columns:
        return df

    enriched = round_end_df.copy()
    if "round" in enriched.columns:
        round_offset = _event_round_offset(round_lookup, enriched)
        enriched["round_number"] = pd.to_numeric(enriched["round"], errors="coerce").astype("float64") + float(round_offset)
    else:
        enriched = add_round_number_from_tick(enriched, round_lookup)

    enriched = enriched.dropna(subset=["round_number"])
    if enriched.empty:
        return df

    enriched["winner_normalized"] = enriched["winner"].astype(str).str.strip().str.upper()
    winner_by_round = (
        enriched.groupby("round_number", dropna=False)["winner_normalized"]
        .agg(get_last_non_null)
        .reset_index()
    )
    winner_map = {"CT": 3, "T": 2}
    winner_by_round["winner_team_num"] = winner_by_round["winner_normalized"].map(winner_map)
    winner_by_round["loser_team_num"] = winner_by_round["winner_team_num"].map({3: 2, 2: 3})
    winner_by_round["winner_team_num_known"] = winner_by_round["winner_team_num"].notna()

    df = df.merge(
        winner_by_round[["round_number", "winner_team_num", "loser_team_num", "winner_team_num_known"]],
        on="round_number",
        how="left",
        suffixes=("", "_event"),
    )
    for column in ("winner_team_num", "loser_team_num", "winner_team_num_known"):
        event_column = f"{column}_event"
        if event_column in df.columns:
            df[column] = df[event_column].where(df[event_column].notna(), df[column])
            df = df.drop(columns=[event_column])

    df["winner_team_num_known"] = df["winner_team_num_known"].fillna(False).astype(bool)
    return df


def add_basic_round_reward_columns(round_events_df: pd.DataFrame) -> pd.DataFrame:
    df = round_events_df.copy()
    df["terminal_reward_ct"] = 0.0
    df["terminal_reward_t"] = 0.0
    df["reward_known"] = False

    if "winner_team_num_known" not in df.columns or "winner_team_num" not in df.columns:
        return df

    ct_mask = (df["winner_team_num_known"] == True) & (df["winner_team_num"] == 3)
    t_mask = (df["winner_team_num_known"] == True) & (df["winner_team_num"] == 2)

    df.loc[ct_mask, ["terminal_reward_ct", "terminal_reward_t", "reward_known"]] = [1.0, -1.0, True]
    df.loc[t_mask, ["terminal_reward_ct", "terminal_reward_t", "reward_known"]] = [-1.0, 1.0, True]
    return df


def build_round_events(raw_df: pd.DataFrame, parsed_events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    round_lookup = build_round_lookup(raw_df)
    grouped = raw_df.groupby("total_rounds_played", dropna=False)

    round_events_df = pd.DataFrame({
        "round_number": list(grouped.groups.keys()),
        "start_tick": grouped["tick"].min().values if "tick" in raw_df.columns else [None] * len(grouped),
        "end_tick": grouped["tick"].max().values if "tick" in raw_df.columns else [None] * len(grouped),
        "start_time": grouped["round_start_time"].min().values if "round_start_time" in raw_df.columns else [None] * len(grouped),
        "end_time": grouped["round_start_time"].max().values if "round_start_time" in raw_df.columns else [None] * len(grouped),
        "round_win_status": grouped["round_win_status"].agg(get_last_non_null).values if "round_win_status" in raw_df.columns else [None] * len(grouped),
        "round_win_reason": grouped["round_win_reason"].agg(get_last_non_null).values if "round_win_reason" in raw_df.columns else [None] * len(grouped),
        "ct_losing_streak": grouped["ct_losing_streak"].agg(get_last_non_null).values if "ct_losing_streak" in raw_df.columns else [None] * len(grouped),
        "t_losing_streak": grouped["t_losing_streak"].agg(get_last_non_null).values if "t_losing_streak" in raw_df.columns else [None] * len(grouped),
        "is_bomb_planted_any": grouped["is_bomb_planted"].agg(lambda series: series.fillna(False).astype(bool).any()).values if "is_bomb_planted" in raw_df.columns else [False] * len(grouped),
    })

    tick_summaries = {
        "bomb_planted": summarize_event_ticks(parsed_events.get("bomb_planted", pd.DataFrame()), round_lookup, "bomb_planted"),
        "bomb_defused": summarize_event_ticks(parsed_events.get("bomb_defused", pd.DataFrame()), round_lookup, "bomb_defused"),
    }
    count_summaries = {
        "bomb_dropped_count": summarize_event_counts(parsed_events.get("bomb_dropped", pd.DataFrame()), round_lookup, "bomb_dropped_count"),
        "bomb_pickup_count": summarize_event_counts(parsed_events.get("bomb_pickup", pd.DataFrame()), round_lookup, "bomb_pickup_count"),
        "kills_count": summarize_event_counts(parsed_events.get("player_death", pd.DataFrame()), round_lookup, "kills_count"),
        "damage_events_count": summarize_event_counts(parsed_events.get("player_hurt", pd.DataFrame()), round_lookup, "damage_events_count"),
    }

    for summary_df in tick_summaries.values():
        round_events_df = round_events_df.merge(summary_df, on="round_number", how="left")
    for summary_df in count_summaries.values():
        round_events_df = round_events_df.merge(summary_df, on="round_number", how="left")

    for column in ("bomb_planted_tick", "bomb_defused_tick"):
        if column not in round_events_df.columns:
            round_events_df[column] = pd.NA

    for column in ("bomb_dropped_count", "bomb_pickup_count", "kills_count", "damage_events_count"):
        if column not in round_events_df.columns:
            round_events_df[column] = 0
        round_events_df[column] = pd.to_numeric(round_events_df[column], errors="coerce").fillna(0).astype(int)

    round_events_df = infer_winner_columns(round_events_df, parsed_events, round_lookup)
    round_events_df = add_basic_round_reward_columns(round_events_df)
    return round_events_df.sort_values("round_number").reset_index(drop=True)


def print_dataset_summary(raw_df: pd.DataFrame, clean_play_df: pd.DataFrame, clean_buy_df: pd.DataFrame) -> None:
    raw_rows = len(raw_df)
    print("Dataset summary:")
    print(f"  live rows: {len(clean_play_df)} ({(len(clean_play_df) / raw_rows):.2%} of raw)" if raw_rows else "  live rows: 0")
    print(f"  buy rows: {len(clean_buy_df)} ({(len(clean_buy_df) / raw_rows):.2%} of raw)" if raw_rows else "  buy rows: 0")

    if clean_play_df.empty:
        print("  clean_play_ticks is empty.")
        return

    def _rate(df: pd.DataFrame, column: str) -> float:
        if column not in df.columns or df.empty:
            return 0.0
        series = df[column]
        if series.dtype == bool:
            return float(series.mean())
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return float((numeric != 0).mean())

    mouse_nonzero = 0.0
    if {"usercmd_mouse_dx", "usercmd_mouse_dy"}.issubset(clean_play_df.columns):
        mouse_dx = pd.to_numeric(clean_play_df["usercmd_mouse_dx"], errors="coerce").fillna(0.0)
        mouse_dy = pd.to_numeric(clean_play_df["usercmd_mouse_dy"], errors="coerce").fillna(0.0)
        mouse_nonzero = float(((mouse_dx != 0.0) | (mouse_dy != 0.0)).mean())

    print(f"  spotted rate: {_rate(clean_play_df, 'spotted'):.2%}")
    print(f"  fire rate: {_rate(clean_play_df, 'FIRE'):.2%}")
    print(f"  nonzero mouse rate: {mouse_nonzero:.2%}")


def parse_events_and_save(parser: DemoParser, demo_stem: str, paths: Paths) -> tuple[dict[str, pd.DataFrame], list[str]]:
    parsed_events: dict[str, pd.DataFrame] = {}
    saved_events: list[str] = []

    for event_name in EVENTS:
        try:
            event_df = parse_single_event(parser, event_name)
            parsed_events[event_name] = event_df
            output_path = paths.events_dir / f"{demo_stem}_{event_name}.parquet"
            event_df.to_parquet(output_path, index=False)
            saved_events.append(event_name)
        except Exception as exc:
            print(f"[warning] Failed to parse/save event '{event_name}': {exc}")
            parsed_events[event_name] = pd.DataFrame()

    return parsed_events, saved_events


def upsert_registry_entry(
    registry: dict[str, Any],
    demo_name: str,
    file_paths: dict[str, Path],
    stats: dict[str, int],
    project_root: Path,
) -> None:
    entry = {
        "demo_name": demo_name,
        "status": "completed",
        "parsed_at": datetime.now().replace(microsecond=0).isoformat(),
        "files": {key: make_registry_relpath(path, project_root) for key, path in file_paths.items()},
        "stats": stats,
    }

    parsed_entries = registry.setdefault("parsed", [])
    for index, existing in enumerate(parsed_entries):
        if isinstance(existing, dict) and existing.get("demo_name") == demo_name:
            parsed_entries[index] = entry
            return

    parsed_entries.append(entry)


def process_demo(demo_path: Path, paths: Paths, registry: dict[str, Any]) -> int:
    demo_stem = demo_path.stem
    output_paths = get_expected_output_paths(paths, demo_path.name)

    parser = DemoParser(str(demo_path))
    raw_df = pd.DataFrame(parser.parse_ticks(TICK_FIELDS))
    if raw_df.empty:
        raise RuntimeError("Raw ticks dataframe is empty.")

    raw_df.to_parquet(output_paths["raw_ticks"], index=False)
    print(f"Raw ticks shape: {raw_df.shape}")
    print(f"Saved raw ticks: {output_paths['raw_ticks']}")

    parsed_events, saved_events = parse_events_and_save(parser, demo_stem, paths)
    print(f"Saved events: {saved_events if saved_events else 'none'}")

    clean_play_df = create_clean_play_ticks(raw_df, parsed_events)
    clean_play_df.to_parquet(output_paths["clean_play_ticks"], index=False)
    print(f"Clean play ticks shape: {clean_play_df.shape}")
    print(f"Saved clean play ticks: {output_paths['clean_play_ticks']}")

    clean_buy_df = create_clean_buy_ticks(raw_df)
    clean_buy_df.to_parquet(output_paths["clean_buy_ticks"], index=False)
    print(f"Clean buy ticks shape: {clean_buy_df.shape}")
    print(f"Saved clean buy ticks: {output_paths['clean_buy_ticks']}")

    round_events_df = build_round_events(raw_df, parsed_events)
    round_events_df.to_parquet(output_paths["round_events"], index=False)
    print(f"Round events shape: {round_events_df.shape}")
    print(f"Saved round events: {output_paths['round_events']}")
    print_dataset_summary(raw_df, clean_play_df, clean_buy_df)

    for key, path in output_paths.items():
        if not path.exists():
            raise RuntimeError(f"Expected output file was not created: {key} -> {path}")

    stats = {
        "raw_rows": int(len(raw_df)),
        "play_rows": int(len(clean_play_df)),
        "buy_rows": int(len(clean_buy_df)),
        "round_count": int(len(round_events_df)),
    }
    upsert_registry_entry(registry, demo_path.name, output_paths, stats, paths.project_root)
    save_registry(paths.registry_file, registry)
    print(f"Registry updated: {paths.registry_file}")
    return 0


def main() -> int:
    args = parse_args()
    paths = build_paths(args.demos_dir, args.dataset_dir)
    ensure_directories(paths)
    registry = load_registry(paths.registry_file)

    demo_files, selected_demo = find_demo_to_process(paths, registry, args.force, args.demo)

    print(f"Found demos: {len(demo_files)}")
    print(f"Registry file: {paths.registry_file}")

    if not demo_files and not args.demo:
        print(f"No .dem files found in {paths.demos_dir}")
        return 0

    if selected_demo is None:
        print("All demos are already completed. Nothing to do.")
        return 0

    print(f"Selected demo: {selected_demo.name}")

    try:
        return process_demo(selected_demo, paths, registry)
    except Exception as exc:
        print(f"[error] Failed to process demo '{selected_demo.name}': {exc}")
        print("Registry was not updated.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
