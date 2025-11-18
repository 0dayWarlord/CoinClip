#seed utilities for reproducibility

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    set random seeds for reproducibility

    Args:
        seed: random seed value
        deterministic: if True, use deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #enable deterministic mode for some operations
        torch.use_deterministic_algorithms(True, warn_only=True)

