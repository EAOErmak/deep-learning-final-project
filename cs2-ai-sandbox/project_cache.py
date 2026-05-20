from __future__ import annotations

import sys
from pathlib import Path


def configure_project_pycache(project_root: Path) -> Path:
    cache_dir = project_root / '.cache' / 'pycache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    if getattr(sys, 'pycache_prefix', None) is None:
        sys.pycache_prefix = str(cache_dir)
    return cache_dir
