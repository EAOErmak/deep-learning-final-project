from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def tensorboard_available() -> bool:
    return SummaryWriter is not None


def sanitize_run_name(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._-')
    return cleaned or 'run'


def create_summary_writer(
    runs_dir: Path,
    run_name: str | None,
    default_prefix: str,
    save_path: Path,
    config: dict[str, Any] | None = None,
) -> tuple[Any | None, Path | None]:
    if SummaryWriter is None:
        return None, None

    resolved_runs_dir = Path(runs_dir)
    resolved_runs_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'{default_prefix}_{sanitize_run_name(save_path.stem)}_{timestamp}'
    else:
        run_name = sanitize_run_name(run_name)

    run_dir = resolved_runs_dir / run_name
    writer = SummaryWriter(log_dir=str(run_dir))

    if config:
        writer.add_text('run/config', json.dumps(config, ensure_ascii=True, indent=2, default=str), 0)

    return writer, run_dir


def log_scalar_dict(
    writer: Any | None,
    prefix: str,
    values: dict[str, Any],
    step: int,
    ignored_keys: set[str] | None = None,
) -> None:
    if writer is None:
        return

    skip_keys = ignored_keys or set()
    for key, value in values.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            writer.add_scalar(f'{prefix}/{key}', int(value), step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(f'{prefix}/{key}', value, step)
        elif isinstance(value, dict):
            log_scalar_dict(writer, f'{prefix}/{key}', value, step, ignored_keys=skip_keys)


def close_summary_writer(writer: Any | None) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()
