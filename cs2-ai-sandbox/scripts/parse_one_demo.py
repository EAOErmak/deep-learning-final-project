from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
    "aim_punch_angle",
    "aim_punch_angle_vel",
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


@dataclass(frozen=True)
class Paths:
    project_root: Path
    demos_dir: Path
    dataset_dir: Path
    registry_file: Path
    raw_ticks_dir: Path
    events_dir: Path


def build_paths() -> Paths:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    return Paths(
        project_root=project_root,
        demos_dir=project_root / "demos",
        dataset_dir=dataset_dir,
        registry_file=dataset_dir / "parsed_demos.json",
        raw_ticks_dir=dataset_dir / "raw_ticks",
        events_dir=dataset_dir / "events",
    )


def ensure_directories(paths: Paths) -> None:
    paths.demos_dir.mkdir(parents=True, exist_ok=True)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_ticks_dir.mkdir(parents=True, exist_ok=True)
    paths.events_dir.mkdir(parents=True, exist_ok=True)

    if not paths.registry_file.exists():
        paths.registry_file.write_text(
            json.dumps({"parsed": []}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )


def load_registry(registry_file: Path) -> dict[str, Any]:
    if not registry_file.exists():
        return {"parsed": []}

    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Registry file is not valid JSON: {registry_file}") from exc

    parsed = data.get("parsed")
    if not isinstance(parsed, list):
        raise RuntimeError(f"Registry format is invalid: {registry_file}")

    return data


def save_registry(registry_file: Path, registry: dict[str, Any]) -> None:
    registry_file.write_text(
        json.dumps(registry, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def find_demo_to_parse(demos_dir: Path, parsed_demo_names: set[str]) -> tuple[list[Path], Path | None]:
    demo_files = sorted(demos_dir.glob("*.dem"))
    selected_demo = next((path for path in demo_files if path.name not in parsed_demo_names), None)
    return demo_files, selected_demo


def parse_ticks(parser: DemoParser, output_path: Path) -> int:
    ticks_df = parser.parse_ticks(TICK_FIELDS)
    ticks_df.to_parquet(output_path, index=False)
    return len(ticks_df.index)


def parse_single_event(parser: DemoParser, event_name: str) -> Any:
    if hasattr(parser, "parse_event"):
        return parser.parse_event(event_name)
    if hasattr(parser, "parse_events"):
        return parser.parse_events(event_name)
    raise AttributeError("demoparser2 parser has no parse_event/parse_events method")


def parse_and_save_events(parser: DemoParser, events_dir: Path, demo_stem: str) -> list[str]:
    saved_events: list[str] = []

    for event_name in EVENTS:
        try:
            event_df = parse_single_event(parser, event_name)
            output_path = events_dir / f"{demo_stem}_{event_name}.parquet"
            event_df.to_parquet(output_path, index=False)
            print(f"[event] Saved '{event_name}' -> {output_path} ({len(event_df.index)} rows)")
            saved_events.append(event_name)
        except Exception as exc:
            print(f"[warning] Failed to parse event '{event_name}': {exc}")

    return saved_events


def append_registry_entry(registry: dict[str, Any], demo_name: str, ticks_file: Path, project_root: Path) -> None:
    registry["parsed"].append(
        {
            "demo_name": demo_name,
            "ticks_file": ticks_file.relative_to(project_root).as_posix(),
            "parsed_at": datetime.now().replace(microsecond=0).isoformat(),
        }
    )


def main() -> int:
    paths = build_paths()
    ensure_directories(paths)

    registry = load_registry(paths.registry_file)
    parsed_demo_names = {
        entry["demo_name"]
        for entry in registry["parsed"]
        if isinstance(entry, dict) and "demo_name" in entry
    }

    demo_files, selected_demo = find_demo_to_parse(paths.demos_dir, parsed_demo_names)

    print(f"Found demos: {len(demo_files)}")
    print(f"Already parsed: {len(parsed_demo_names)}")

    if not demo_files:
        print(f"No .dem files found in {paths.demos_dir}")
        return 0

    if selected_demo is None:
        print("All demos are already parsed. Nothing to do.")
        return 0

    print(f"Selected demo: {selected_demo.name}")

    ticks_output = paths.raw_ticks_dir / f"{selected_demo.stem}_ticks.parquet"

    try:
        parser = DemoParser(str(selected_demo))
        tick_rows = parse_ticks(parser, ticks_output)
        print(f"Ticks rows: {tick_rows}")
        print(f"Ticks saved to: {ticks_output}")

        saved_events = parse_and_save_events(parser, paths.events_dir, selected_demo.stem)
        print(f"Saved events: {saved_events if saved_events else 'none'}")

        append_registry_entry(registry, selected_demo.name, ticks_output, paths.project_root)
        save_registry(paths.registry_file, registry)
        print(f"Registry updated: {paths.registry_file}")
        print(f"Demo added to registry: {selected_demo.name}")
    except Exception as exc:
        print(f"[error] Failed to parse demo '{selected_demo.name}': {exc}")
        print("Registry was not updated.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
