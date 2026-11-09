"""Reject unsupported trainable inputs before scalar or NumPy conversion."""
from collections.abc import Mapping

import numpy as np
import torch


def require_fixed_input(value, *, name):
    """Inspect input metadata without reading tensor values or allocating fields."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise NotImplementedError(f"Trainable {name} is not supported by this constructor; provide a fixed value.")
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            require_fixed_input(item, name=name)
    elif isinstance(value, Mapping):
        for item in value.values():
            require_fixed_input(item, name=name)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_fixed_input(item, name=name)
