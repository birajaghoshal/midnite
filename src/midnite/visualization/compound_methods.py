"""Compound methods created from building block, implementing common use cases."""
from typing import List

import torch
from torch import Tensor
from torch.nn import functional
from torch.nn.modules import Module
from torch.nn.modules import Sequential

import midnite
from midnite.visualization.base import BilateralTransform
from midnite.visualization.base import GradAM
from midnite.visualization.base import GuidedBackpropagation
from midnite.visualization.base import NeuronSelector
from midnite.visualization.base import PixelActivation
from midnite.visualization.base import RandomTransform
from midnite.visualization.base import ResizeTransform
from midnite.visualization.base import SimpleSelector
from midnite.visualization.base import SpatialSplit
from midnite.visualization.base import SplitSelector
from midnite.visualization.base import TVRegularization
from midnite.visualization.base import WeightDecay


def _prepare_input(img: Tensor) -> Tensor:
    return img.clone().detach().to(midnite.get_device())


def _top_k_selector(out: Tensor, k: int = 3) -> NeuronSelector:
    if k < 1:
        raise ValueError(f"At least one class required. Got: {k}")
    mask = torch.zeros_like(out, device=midnite.get_device())
    classes = out.topk(dim=0, k=k)[1]
    for i in range(k):
        mask[classes[i]] = 1

    return SimpleSelector(mask)


def saliency_map(model: Module, input_image: Tensor, n_top_classes: int = 3):
    """Calculates a simple saliency map of the pixel influence on the top n classes.

    Args:
        model: the whole model
        input_image: image tensor of dimensions (c, h, w)
        n_top_classes: the number of classes to take into account

    Returns:
        a saliency heatmap of dimensions (h, w)

    """
    # Find classes to analyze
    input_image = _prepare_input(input_image)
    res = model.to(midnite.get_device)(input_image.unsqueeze(dim=0))
    selector = _top_k_selector(res, n_top_classes)

    # Apply guided backpropagation
    backprop = GuidedBackpropagation([model], selector, SpatialSplit())
    return backprop.visualize(input_image)


def gradcam(
    features: List[Module],
    classifier: List[Module],
    input_image: Tensor,
    n_top_classes: int = 3,
):
    """Performs GradCAM on an input image.

    For further explanations about GradCam, see https://arxiv.org/abs/1610.02391.

    Args:
        features: the spatial feature layers of the model
        classifier: the classifier layers of the model
        input_image: image tensor of dimensions (c, h, w)
        rescale: whether to upsample the result to input dimensions
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a GradCAM heatmap of dimensions (h, w)

    """
    # Prepare model and input
    model = Sequential(*features, *classifier)
    model.to(midnite.get_device()).eval()
    input_image = _prepare_input(input_image)

    out = model(input_image.unsqueeze(0)).squeeze(0)

    # Get top N classes
    selector = _top_k_selector(out, n_top_classes)

    # Apply GradAM on Classes
    gradcam = GradAM(classifier, selector, features, SpatialSplit())
    result = gradcam.visualize(input_image)

    return (
        functional.interpolate(
            result.unsqueeze(0).unsqueeze(0),
            size=tuple(input_image.size()[1:]),
            mode="bilinear",
            align_corners=True,
        )
        .squeeze(0)
        .squeeze(0)
    )


def class_visualization(model: Module, class_index: int):
    """Visualizes a class for a classification network.

    Args:
        model: the model to visualize for
        class_index: the index of the class to visualize

    """

    img = PixelActivation(
        [model],
        SplitSelector(SpatialSplit(), [class_index]),
        opt_n=500,
        iter_n=20,
        init_size=50,
        transform=RandomTransform(scale_fac=0)
        + BilateralTransform(1.1)
        + ResizeTransform(),
        regularization=[TVRegularization(0.05), WeightDecay(1e-12)],
    ).visualize()

    return PixelActivation(
        [model],
        SplitSelector(SpatialSplit(), [class_index]),
        opt_n=100,
        iter_n=int(50),
        transform=RandomTransform() + BilateralTransform(),
        regularization=[TVRegularization(0.1), WeightDecay(1e-10)],
    ).visualize(img)
