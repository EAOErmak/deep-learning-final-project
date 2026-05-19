from pathlib import Path
import random
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    Console = None


RAW_TICKS_DIR = Path("dataset/raw_ticks")
SAMPLE_COUNT = 10

IMPORTANT_COLUMNS = [
    "tick",
    "name",
    "player_name",
    "steamid",
    "X", "Y", "Z",
    "velocity_X", "velocity_Y", "velocity_Z",
    "health",
    "armor_value",
    "has_helmet",
    "is_alive",
    "team_num",
    "balance",
    "active_weapon_name",
    "active_weapon_ammo",
    "total_ammo_left",
    "pitch",
    "yaw",
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "FIRE",
    "WALK",
    "USE",
    "is_bomb_planted",
    "is_freeze_period",
    "round_in_progress",
    "total_rounds_played",
]


def find_parquet_files() -> list[Path]:
    return sorted(RAW_TICKS_DIR.glob("*.parquet"))


def pick_random_rows(files: list[Path], count: int) -> pd.DataFrame:
    samples = []

    attempts = 0
    max_attempts = count * 5

    while len(samples) < count and attempts < max_attempts:
        attempts += 1

        file = random.choice(files)

        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"[WARN] Could not read {file}: {e}")
            continue

        if df.empty:
            continue

        row = df.sample(n=1).copy()
        row.insert(0, "source_file", file.name)
        samples.append(row)

    if not samples:
        return pd.DataFrame()

    return pd.concat(samples, ignore_index=True)


def select_pretty_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = ["source_file"] + [c for c in IMPORTANT_COLUMNS if c in df.columns]
    result = df[existing].copy()

    for col in ["X", "Y", "Z", "velocity_X", "velocity_Y", "velocity_Z", "pitch", "yaw"]:
        if col in result.columns:
            result[col] = result[col].map(lambda v: round(float(v), 2) if pd.notna(v) else v)

    return result


def print_with_rich(df: pd.DataFrame) -> None:
    console = Console()

    console.print(
        Panel.fit(
            f"Random dataset samples: [bold]{len(df)}[/bold]\n"
            f"Dataset path: [cyan]{RAW_TICKS_DIR}[/cyan]",
            title="CS2 Demo Dataset Viewer",
        )
    )

    for i, row in df.iterrows():
        table = Table(title=f"Sample #{i + 1}", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="bold")
        table.add_column("Value")

        for col, value in row.items():
            table.add_row(str(col), str(value))

        console.print(table)


def print_fallback(df: pd.DataFrame) -> None:
    print("\n=== Random Dataset Samples ===\n")

    for i, row in df.iterrows():
        print(f"\n--- Sample #{i + 1} ---")
        for col, value in row.items():
            print(f"{col:25} {value}")


def main() -> None:
    files = find_parquet_files()

    if not files:
        print(f"No parquet files found in: {RAW_TICKS_DIR}")
        return

    print(f"Found parquet files: {len(files)}")

    samples = pick_random_rows(files, SAMPLE_COUNT)

    if samples.empty:
        print("Could not collect random samples.")
        return

    pretty_samples = select_pretty_columns(samples)

    if Console is not None:
        print_with_rich(pretty_samples)
    else:
        print_fallback(pretty_samples)
        print("\nTip: install rich for prettier output:")
        print("pip install rich")


if __name__ == "__main__":
    main()