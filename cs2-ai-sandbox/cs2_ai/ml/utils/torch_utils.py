from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except Exception:
    torch = None


def torch_available() -> bool:
    return torch is not None


def get_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_num_workers(requested_workers: int | None) -> int:
    if requested_workers is not None and requested_workers >= 0:
        return requested_workers
    cpu_count = os.cpu_count() or 1
    if os.name == "nt":
        # Windows multiprocessing copies the dataset object into each worker.
        # These training datasets keep sizable demo/state caches, so high worker
        # counts explode RAM usage much faster than on Linux.
        return max(1, min(2, cpu_count - 1))
    return max(1, min(8, cpu_count - 1))


def configure_torch_runtime(device: str) -> dict[str, object]:
    runtime_info: dict[str, object] = {
        "device": device,
        "matmul_precision": None,
        "cudnn_benchmark": False,
        "tf32": False,
    }
    if torch is None:
        return runtime_info

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        runtime_info["matmul_precision"] = "high"

    if device == "cuda" and torch.cuda.is_available():
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
            runtime_info["cudnn_benchmark"] = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        runtime_info["tf32"] = True

    return runtime_info


def build_dataloader_kwargs(
    device: str,
    num_workers: int | None,
    *,
    is_training: bool,
) -> dict[str, object]:
    resolved_workers = resolve_num_workers(num_workers)
    if os.name == "nt":
        if not is_training:
            # On Windows, spawning a second multiprocessing DataLoader for validation
            # after a training loader is active is brittle and can fail with worker exits.
            resolved_workers = 0
        kwargs: dict[str, object] = {
            "num_workers": resolved_workers,
            "pin_memory": device == "cuda",
        }
        if resolved_workers > 0:
            # Keep the Windows loader conservative to avoid multiplying dataset caches.
            kwargs["persistent_workers"] = False
            kwargs["prefetch_factor"] = 1
        return kwargs
    kwargs: dict[str, object] = {
        "num_workers": resolved_workers,
        "pin_memory": device == "cuda",
    }
    if resolved_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4 if is_training else 2
    return kwargs
