from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
from pathlib import Path

from cs2_ai.ml.reporting import DEFAULT_REPORTS_DIR, flatten_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize training experiment reports')
    parser.add_argument('--reports-dir', type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--module', type=str, default=None)
    parser.add_argument('--export-csv', type=Path, default=None)
    return parser.parse_args()


def load_reports(reports_dir: Path) -> list[dict[str, object]]:
    if not reports_dir.exists():
        return []
    reports: list[dict[str, object]] = []
    for path in sorted(reports_dir.glob('*.json')):
        try:
            reports.append(json.loads(path.read_text(encoding='utf-8')))
        except Exception as exc:
            print(f'Skipping invalid report {path}: {exc}')
    return reports


def sort_key(report: dict[str, object]) -> tuple[float, str]:
    loss = report.get('val_loss')
    try:
        loss_value = float(loss)
    except Exception:
        loss_value = float('inf')
    return (loss_value, str(report.get('run_id', '')))


def print_table(rows: list[dict[str, object]]) -> None:
    headers = ['module', 'model', 'val_loss', 'train_loss', 'seq_len', 'chunk_len', 'checkpoint']
    widths = {header: len(header) for header in headers}
    rendered_rows: list[dict[str, str]] = []
    for row in rows:
        rendered = {
            'module': str(row.get('module_name', '')),
            'model': str(row.get('model_name', '')),
            'val_loss': f"{float(row.get('val_loss', 0.0)):.4f}" if row.get('val_loss') is not None else 'n/a',
            'train_loss': f"{float(row.get('train_loss', 0.0)):.4f}" if row.get('train_loss') is not None else 'n/a',
            'seq_len': str(row.get('seq_len', '')),
            'chunk_len': str(row.get('chunk_len', '')),
            'checkpoint': str(row.get('checkpoint_path', '')),
        }
        rendered_rows.append(rendered)
        for key, value in rendered.items():
            widths[key] = max(widths[key], len(value))
    header_line = ' | '.join(header.ljust(widths[header]) for header in headers)
    separator = '-+-'.join('-' * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rendered_rows:
        print(' | '.join(row[header].ljust(widths[header]) for header in headers))


def export_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    flat_rows = [flatten_metrics(row) for row in rows]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in flat_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    reports = load_reports(args.reports_dir)
    if args.module:
        reports = [report for report in reports if str(report.get('module_name')) == args.module]
    if not reports:
        print(f'No reports found in {args.reports_dir}')
        return 1

    grouped: dict[str, list[dict[str, object]]] = {}
    for report in reports:
        grouped.setdefault(str(report.get('module_name', 'unknown')), []).append(report)

    best_rows: list[dict[str, object]] = []
    for module_name in sorted(grouped):
        module_reports = sorted(grouped[module_name], key=sort_key)[: max(1, args.top_k)]
        print(f'\n[{module_name}] top {len(module_reports)}')
        print_table(module_reports)
        best_rows.extend(module_reports)

    if args.export_csv is not None:
        export_csv(best_rows, args.export_csv)
        print(f'\nExported summary CSV: {args.export_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
