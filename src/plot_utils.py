import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import squeeze
from torch import Tensor


def show(img: Tensor, scale: float = 1.0):
    """Shows an torch tensor as image using pyplot.

     Fixes image dimensions, if necessary.

    Args:
        img: the tensor to show
        scale: the scale of the plot to show

    """
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
                img = img.permute(1, 2, 0)
            elif img.size(dim=2) == 1:
                img = img.squeeze(dim=2)
            else:
                raise ValueError("Invalid number of channels for image")
    # Also not image without channels
    elif not len(img.size()) == 2:
        raise ValueError("Invalid dimensions for image")

    # Transfer to cpu and plot
    fix, ax = plt.subplots(figsize=(8 * scale, 6 * scale))
    ax.grid(False)
    ax.imshow(img.cpu())
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


def plot_saliency(
    saliency: Tensor, img: Image, sel_layer: int, output_layer, plot_with_image=True
):
    """Plots the saliency map.

     Args:
        saliency: tensor of class activations with dimensions mini-batch x channels x height x width
        img: input image
        sel_layer: layer number of selected layer
        output_layer: layer number of target layer
        plot_with_image: plot saliency together with image (True) or alone (False)

    """
    if not len(saliency.size()) == 4:
        raise ValueError("saliency has to be 4 dimensions, got: ", len(saliency.size()))
    saliency = squeeze(saliency)
    if not len(saliency.size()) == 2:
        raise ValueError(
            "saliency has to have 2 dimensions with more than one element, got: ",
            len(saliency.size()),
        )
    sal_img = Image.fromarray(saliency.numpy())
    sal_img = sal_img.resize(img.size, resample=Image.LINEAR)

    plt.title(
        "Activation map for layer: {} "
        "with respect to layer: {}".format(sel_layer, output_layer)
    )

    if plot_with_image:
        plt.imshow(img)
    plt.imshow(np.array(sal_img), alpha=0.5, cmap="jet")

    plt.show()
