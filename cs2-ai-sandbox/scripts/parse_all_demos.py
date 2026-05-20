from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import argparse

try:
    from parse_and_clean_one_demo import (
        build_paths,
        ensure_directories,
        find_demo_to_process,
        load_registry,
        process_demo,
    )
except ModuleNotFoundError:
    from scripts.parse_and_clean_one_demo import (
        build_paths,
        ensure_directories,
        find_demo_to_process,
        load_registry,
        process_demo,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parse all CS2 demos that are not yet completed in parsed_demos.json.'
    )
    parser.add_argument('--force', action='store_true', help='Reparse demos even if they are already completed.')
    parser.add_argument('--demos-dir', type=str, default='demos', help='Directory containing .dem files.')
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Directory for output parquet files.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = build_paths(args.demos_dir, args.dataset_dir)
    ensure_directories(paths)

    processed_count = 0
    failed_count = 0

    while True:
        registry = load_registry(paths.registry_file)
        demo_files, selected_demo = find_demo_to_process(paths, registry, args.force, demo_arg=None)

        print(f'Found demos: {len(demo_files)}')
        print(f'Registry file: {paths.registry_file}')

        if not demo_files:
            print(f'No .dem files found in {paths.demos_dir}')
            break

        if selected_demo is None:
            print('All demos are already completed. Nothing to do.')
            break

        print(f'Selected demo: {selected_demo.name}')

        try:
            process_demo(selected_demo, paths, registry)
            processed_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"[error] Failed to process demo '{selected_demo.name}': {exc}")
            print('Registry was not updated for this demo.')
            break

    print(f'Finished. Processed={processed_count}, Failed={failed_count}')
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
