import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import squeeze
from torch import Tensor


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
    sal_map = Image.fromarray(saliency.numpy())
    sal_map = sal_map.resize(img.size, resample=Image.LINEAR)

    plt.title(
        "Activation map for layer: {} "
        "with respect to layer: {}".format(sel_layer, output_layer)
    )

    if plot_with_image:
        plt.imshow(img)
    plt.imshow(np.array(sal_map), alpha=0.5, cmap="jet")

    plt.show()


def plot_guided_backprop(saliency: Tensor, img):
    """plots the guided gradients in the input image dimensions.
    Args:
        saliency: tensor of class activations with dimensions mini-batch x channels x height x width
        img: input image
    """

    saliency = squeeze(saliency)
    if not len(saliency.size()) == 3:
        raise ValueError(
            "saliency has to have 3 dimensions with more than one element, got: ",
            len(saliency.size()),
        )

    saliency = saliency.mean(0)
    sal_map = Image.fromarray(saliency.numpy())
    sal_map = sal_map.resize(img.size, resample=Image.LINEAR)
    plt.imshow(sal_map)
    plt.show()
