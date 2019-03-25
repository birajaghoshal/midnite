"""General utilities."""
from typing import Tuple

import torch
from torch import device
from torch import dtype


def tensor_defaults() -> Tuple[device, dtype]:
    """Checks the device and type of newly created tensors

    Returns:
        (default device, default type)

    """
    default = torch.tensor([])
    device = "cuda" + str(torch.cuda.current_device()) if default.is_cuda else "cpu"
    return torch.device(device), default.dtype
