from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / 'artifacts' / 'reports'


def get_git_commit_hash(project_root: Path | None = None) -> str | None:
    root = Path(project_root or PROJECT_ROOT)
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    value = (result.stdout or '').strip()
    return value or None


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, set):
        return sorted(sanitize_for_json(item) for item in value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def flatten_metrics(payload: dict[str, Any], prefix: str = '') -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, full_key))
        else:
            flat[full_key] = sanitize_for_json(value)
    return flat


def build_run_id(module_name: str, model_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{timestamp}_{module_name}_{model_name}'


def build_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Training Report: {report['module_name']}",
        '',
        f"- Run id: `{report['run_id']}`",
        f"- Model: `{report['model_name']}`",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Split mode: `{report['split_mode']}`",
        f"- Seq len: `{report['seq_len']}`",
        f"- Feature dim: `{report['feature_dim']}`",
        f"- Target shape: `{report['target_shape']}`",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Git commit: `{report.get('git_commit_hash') or 'unknown'}`",
        '',
        '## Metrics',
        '',
        f"- Train loss: `{report.get('train_loss')}`",
        f"- Val loss: `{report.get('val_loss')}`",
        '',
        '## Train Metrics',
        '```json',
        json.dumps(report.get('train_metrics', {}), indent=2, ensure_ascii=True),
        '```',
        '',
        '## Val Metrics',
        '```json',
        json.dumps(report.get('val_metrics', {}), indent=2, ensure_ascii=True),
        '```',
        '',
        '## Config',
        '```json',
        json.dumps(report.get('config', {}), indent=2, ensure_ascii=True),
        '```',
    ]
    return '\n'.join(lines) + '\n'


def write_training_report(report: dict[str, Any], reports_dir: Path | None = None) -> dict[str, str]:
    output_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = sanitize_for_json(report)
    run_id = str(serializable['run_id'])
    json_path = output_dir / f'{run_id}.json'
    csv_path = output_dir / f'{run_id}.csv'
    md_path = output_dir / f'{run_id}.md'

    json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True), encoding='utf-8')

    flat_row = flatten_metrics(serializable)
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)

    md_path.write_text(build_markdown_report(serializable), encoding='utf-8')
    return {
        'json': str(json_path),
        'csv': str(csv_path),
        'markdown': str(md_path),
    }


def build_base_training_report(
    *,
    module_name: str,
    model_name: str,
    dataset_path: str,
    split_mode: str,
    seq_len: int,
    feature_dim: int,
    target_shape: str,
    checkpoint_path: str,
    config: dict[str, Any],
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    chunk_len: int | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    run_id = build_run_id(module_name, model_name)
    report = {
        'run_id': run_id,
        'module_name': module_name,
        'model_name': model_name,
        'dataset_path': dataset_path,
        'split_mode': split_mode,
        'seq_len': int(seq_len),
        'chunk_len': None if chunk_len is None else int(chunk_len),
        'feature_dim': int(feature_dim),
        'target_shape': str(target_shape),
        'checkpoint_path': checkpoint_path,
        'git_commit_hash': get_git_commit_hash(PROJECT_ROOT),
        'config': sanitize_for_json(config),
        'train_loss': sanitize_for_json(train_metrics.get('loss')),
        'val_loss': sanitize_for_json(val_metrics.get('loss')),
        'train_metrics': sanitize_for_json(train_metrics),
        'val_metrics': sanitize_for_json(val_metrics),
        'reports_dir': str(reports_dir or DEFAULT_REPORTS_DIR),
    }
    return report
