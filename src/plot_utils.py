from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import squeeze
from torch import Tensor


def plot_saliency(
    saliency: Tensor,
    img: Image,
    output_class: int,
    output_score: int,
    sel_layer: List[int],
    plot_with_image=True,
):
    """Plots the saliency map.

     Args:
        saliency: tensor of class activations with dimensions mini-batch x channels x height x width
        img: input image
        output_class: classification of the image
        output_score: score of class prediction
        sel_layer: layer for which saliency was computed
        plot_with_image: plot saliency together with image (True) or alone (False)

    """
    saliency = squeeze(saliency)
    sal_map = Image.fromarray(saliency.numpy())
    sal_map = sal_map.resize(img.size, resample=Image.LINEAR)

    plt.title(
        "Activation map for class {} \n "
        "predicted with {:.1f}% \n "
        "for layers: {}".format(output_class, 100 * float(output_score), sel_layer)
    )
    if plot_with_image:
        plt.imshow(img)
    plt.imshow(np.array(sal_map), alpha=0.5, cmap="jet")

    plt.show()
