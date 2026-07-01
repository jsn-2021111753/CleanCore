"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False, torch_num_threads: Optional[int] = None) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

