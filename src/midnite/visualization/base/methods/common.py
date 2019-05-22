"""Some common functionality for visualization methods."""
from torch import Tensor

from midnite.visualization.base import NeuronSelector


def _calculate_single_mean(out, select: NeuronSelector) -> Tensor:
    """Calculates the selected mean for a minibatch with one element, retaining grads.

    Args:
        out: network output of shape (1, c, h, w)
        select: selector

    Returns:
        the calculated mean as scalar tensor

    """
    out = out.squeeze(dim=0)
    mask = select.get_mask(list(out.size()))
    return (out * mask).sum() / mask.sum()
