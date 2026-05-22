from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cs2_ai.dataset.round_identity import round_metadata_from_sample


def _simple_rounds_path(base_path: Path, module_name: str) -> Path:
    return base_path.parent / f'{module_name}_trained_rounds.txt'


def _resolve_dataset_and_indices(dataset_or_subset) -> tuple[Any, list[int]]:
    if hasattr(dataset_or_subset, 'indices') and hasattr(dataset_or_subset, 'dataset'):
        return dataset_or_subset.dataset, [int(idx) for idx in dataset_or_subset.indices]
    return dataset_or_subset, list(range(len(dataset_or_subset)))


def collect_round_usage(dataset_or_subset) -> list[dict[str, object]]:
    dataset, indices = _resolve_dataset_and_indices(dataset_or_subset)
    aggregated: dict[str, dict[str, object]] = {}
    for idx in indices:
        sample_metadata = dataset.get_sample_metadata(int(idx))
        round_meta = round_metadata_from_sample(sample_metadata)
        round_uid = str(round_meta['round_uid'])
        tick_indices = [int(tick) for tick in sample_metadata.get('tick_indices', ())]
        tick_values = tick_indices + [int(sample_metadata.get('target_tick', tick_indices[-1] if tick_indices else 0))]
        entry = aggregated.setdefault(
            round_uid,
            {
                **round_meta,
                'sample_count': 0,
                'tick_min': min(tick_values) if tick_values else 0,
                'tick_max': max(tick_values) if tick_values else 0,
                'perspective_steamids': set(),
            },
        )
        entry['sample_count'] = int(entry['sample_count']) + 1
        if tick_values:
            entry['tick_min'] = min(int(entry['tick_min']), min(tick_values))
            entry['tick_max'] = max(int(entry['tick_max']), max(tick_values))
        perspective = sample_metadata.get('perspective_steamid')
        if perspective is not None:
            entry['perspective_steamids'].add(int(perspective))
    result: list[dict[str, object]] = []
    for round_uid, entry in sorted(aggregated.items()):
        result.append(
            {
                'round_uid': round_uid,
                'demo_name': entry['demo_name'],
                'demo_dir': entry['demo_dir'],
                'round_number': int(entry['round_number']),
                'round_file': entry['round_file'],
                'source_file': entry['source_file'],
                'sample_count': int(entry['sample_count']),
                'tick_min': int(entry['tick_min']),
                'tick_max': int(entry['tick_max']),
                'perspective_count': len(entry['perspective_steamids']),
            }
        )
    return result


@dataclass
class TrainingRoundLedger:
    path: Path
    entries: list[dict[str, object]]

    @classmethod
    def load(cls, path: Path) -> 'TrainingRoundLedger':
        entries: list[dict[str, object]] = []
        resolved = Path(path)
        if resolved.exists():
            for line in resolved.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        return cls(path=resolved, entries=entries)

    def append_run_rounds(
        self,
        *,
        run_id: str,
        module_name: str,
        model_name: str,
        checkpoint_path: str,
        dataset_dir: str,
        dataset_subdir: str,
        split_mode: str,
        split: str,
        round_usage: list[dict[str, object]],
        config_hash: str | None = None,
        git_commit: str | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(timezone.utc).isoformat()
        simple_rounds_path = _simple_rounds_path(self.path, module_name)
        simple_round_uids = sorted({str(item['round_uid']) for item in round_usage})
        with self.path.open('a', encoding='utf-8') as handle:
            for item in round_usage:
                record = {
                    'created_at': created_at,
                    'run_id': run_id,
                    'module_name': module_name,
                    'model_name': model_name,
                    'checkpoint_path': checkpoint_path,
                    'dataset_dir': dataset_dir,
                    'dataset_subdir': dataset_subdir,
                    'split_mode': split_mode,
                    'split': split,
                    'round_uid': item['round_uid'],
                    'demo_name': item['demo_name'],
                    'demo_dir': item['demo_dir'],
                    'round_number': item['round_number'],
                    'round_file': item['round_file'],
                    'source_file': item['source_file'],
                    'sample_count': item['sample_count'],
                    'tick_min': item['tick_min'],
                    'tick_max': item['tick_max'],
                    'perspective_count': item['perspective_count'],
                    'config_hash': config_hash,
                    'git_commit': git_commit,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + '\n')
                self.entries.append(record)
        if simple_round_uids:
            existing: set[str] = set()
            if simple_rounds_path.exists():
                existing = {line.strip() for line in simple_rounds_path.read_text(encoding='utf-8').splitlines() if line.strip()}
            merged = sorted(existing | set(simple_round_uids))
            simple_rounds_path.write_text('\n'.join(merged) + '\n', encoding='utf-8')

    def read_trained_round_uids(
        self,
        *,
        module_name: str | None = None,
        model_name: str | None = None,
        checkpoint_path: str | None = None,
        match_mode: str = 'module',
    ) -> set[str]:
        match_mode = str(match_mode).strip().lower()
        result: set[str] = set()
        if match_mode == 'module' and module_name is not None:
            simple_rounds_path = _simple_rounds_path(self.path, module_name)
            if simple_rounds_path.exists():
                return {line.strip() for line in simple_rounds_path.read_text(encoding='utf-8').splitlines() if line.strip()}
        for entry in self.entries:
            if match_mode == 'module':
                if module_name is not None and entry.get('module_name') == module_name:
                    result.add(str(entry['round_uid']))
            elif match_mode == 'model':
                if module_name is not None and model_name is not None and entry.get('module_name') == module_name and entry.get('model_name') == model_name:
                    result.add(str(entry['round_uid']))
            elif match_mode == 'checkpoint':
                if checkpoint_path is not None and entry.get('checkpoint_path') == checkpoint_path:
                    result.add(str(entry['round_uid']))
            elif match_mode == 'any':
                result.add(str(entry['round_uid']))
            else:
                raise ValueError(f'Unsupported ledger match mode: {match_mode}')
        return result

    def summarize_dataset_rounds(self) -> dict[str, object]:
        unique_rounds = {str(entry['round_uid']) for entry in self.entries}
        modules: dict[str, int] = {}
        models: dict[str, int] = {}
        checkpoints: set[str] = set()
        run_ids: set[str] = set()
        for entry in self.entries:
            modules[str(entry['module_name'])] = modules.get(str(entry['module_name']), 0) + 1
            models[str(entry['model_name'])] = models.get(str(entry['model_name']), 0) + 1
            checkpoints.add(str(entry['checkpoint_path']))
            run_ids.add(str(entry['run_id']))
        return {
            'total_entries': len(self.entries),
            'total_runs': len(run_ids),
            'total_unique_rounds': len(unique_rounds),
            'rounds_per_module': modules,
            'rounds_per_model': models,
            'checkpoints_seen': sorted(checkpoints),
        }
