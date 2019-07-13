"""Utils for plotting in notebooks."""
from typing import Optional

import matplotlib.pyplot as plt
from torch import Tensor

from midnite.visualization.compound_methods import _upscale


def _fix_dims(img: Tensor):
    # Detach from gradient
    img = img.detach()
    # Fix minibatch dimension if necessary
    if img.size(dim=0) == 1:
        img = img.squeeze(dim=0)

    # Image with color channels
    if len(img.size()) == 3:
        if not img.size(dim=2) == 3:
            # Permute or squeeze if necessary
            if img.size(dim=0) == 3:
                img = img.permute((1, 2, 0))
            elif img.size(dim=2) == 1:
                img = img.squeeze(dim=2)
            else:
                raise ValueError("Invalid number of channels for image")
    # Also not image without channels
    elif not len(img.size()) == 2:
        raise ValueError("Invalid dimensions for image")
    return img.cpu()


def show(img: Tensor, scale: float = 1.0):
    """Shows an torch tensor as image using pyplot.

     Fixes image dimensions, if necessary.

    Args:
        img: the tensor to show
        scale: the scale of the plot to show

    """
    img = _fix_dims(img)
    # Transfer to cpu and plot
    fix, ax = plt.subplots(figsize=(8 * scale, 6 * scale))
    ax.grid(False)
    ax.imshow(img)
    plt.show()


def show_normalized(img: Tensor, scale: float = 1.0):
    """Shows the normalized version of a torch tensor as image using pyplot.

    Fixes image dimensions, if necessary.

    Args:
        img: the tensor to show
        scale: the scale of the plot to show

    """
    img = img - img.min()
    img.div_(img.max())
    show(img, scale)


def show_heatmap(heatmap: Tensor, scale: float = 1.0, img: Optional[Tensor] = None):
    """Shows a heatmap.

    Args:
        heatmap: the heatmap to show
        scale: the scale of the image
        img: optional image to overlay

    """
    heatmap = _fix_dims(heatmap)
    fig, ax = plt.subplots(figsize=(8 * scale, 6 * scale))
    ax.grid(False)
    if img is not None:
        img = _fix_dims(img)
        if not img.size()[:2] == heatmap.size()[:2]:
            heatmap = _upscale(heatmap, img.size()[:2])
        im = ax.imshow(heatmap, cmap="jet")
        ax.imshow(img, alpha=0.5)
    else:
        im = ax.imshow(heatmap, cmap="jet")
    fig.colorbar(im, ax=ax)
    plt.show()
