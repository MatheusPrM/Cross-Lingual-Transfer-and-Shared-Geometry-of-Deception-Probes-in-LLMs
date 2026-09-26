"""
Reproducibility utilities.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch.

    If deterministic=True, enables deterministic behavior where possible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True