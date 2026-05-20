from __future__ import annotations

import sys
from pathlib import Path


def _configure_pycache_prefix() -> None:
    if getattr(sys, 'pycache_prefix', None):
        return
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / '.cache' / 'pycache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = str(cache_dir)


_configure_pycache_prefix()
