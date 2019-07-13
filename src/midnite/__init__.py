"""
Global context for controlling computation device.

While generally global state should be avoided, we believe it is correct in this
specific case - since 1) the computation device really _is_ global state of any program
running with pytorch, and 2) this state does not effect any visible branching logic.

On the contrary, making the computation device consistent avoids the frequent errors of
tensors being on different devices.

"""
import threading
from contextlib import contextmanager

import torch

# Root context is cpu device
_device_context = threading.local()
_device_context.stack = [torch.device("cpu")]


@contextmanager
def device(name: str):
    """Creates a device context for midnite to run in.

    Args:
        name: pytorch name of the device, e.g. "cpu" or "cuda:0"

    """
    _device_context.stack.append(torch.device(name))
    yield
    _device_context.stack.pop()


def get_device() -> torch.device:
    """Gets the current torch device.

    Returns: the torch device that is currently active

    """
    return _device_context.stack[-1]
