"""Compound methods created from building block, implementing common use cases."""
from typing import List
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional
from torch.nn.modules import Module
from torch.nn.modules import Sequential

import midnite
from .base import BilateralTransform
from .base import GradAM
from .base import GuidedBackpropagation
from .base import NeuronSelector
from .base import NeuronSplit
from .base import PixelActivation
from .base import RandomTransform
from .base import ResizeTransform
from .base import SimpleSelector
from .base import SpatialSplit
from .base import SplitSelector
from .base import TVRegularization
from .base import WeightDecay


def _prepare_input(img: Tensor) -> Tensor:
    if not len(img.size()) == 3 or not img.size(0) == 3:
        raise ValueError(f"Not an image. Size: {img.size()}")
    return img.clone().detach().to(midnite.get_device())


def _top_k_selector(out: Tensor, k: int = 3) -> NeuronSelector:
    if not len(out.size()) == 2 or not out.size(0) == 1:
        raise ValueError(f"Not a valid output for one input. Size: {out.size()}")
    if k < 1:
        raise ValueError(f"At least one class required. Got: {k}")
    out = out.squeeze(0)
    mask = torch.zeros_like(out, device=midnite.get_device())
    classes = out.topk(dim=0, k=k)[1]
    for i in range(k):
        mask[classes[i]] = 1

    return SimpleSelector(mask)


def _upscale(img: Tensor, size: Tuple[int, int]) -> Tensor:
    return (
        functional.interpolate(
            img.unsqueeze(dim=0).unsqueeze(dim=0),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        .squeeze(0)
        .squeeze(0)
    )


def guided_backpropagation(
    layers: List[Module], input_image: Tensor, n_top_classes: int = 3
) -> Tensor:
    """Calculates a simple saliency map of the pixel influence on the top n classes.

    Args:
        layers: the whole model in its layers
        input_image: image tensor of dimensions (c, h, w)
        n_top_classes: the number of classes to take into account

    Returns:
        a saliency heatmap of dimensions (h, w)

    """
    # Find classes to analyze
    input_image = _prepare_input(input_image).unsqueeze(dim=0)
    net = Sequential(*layers)
    res = (net.to(midnite.get_device()))(input_image)
    selector = _top_k_selector(res, n_top_classes)

    # Apply guided backpropagation
    backprop = GuidedBackpropagation(layers, selector, SpatialSplit())
    return backprop.visualize(input_image)


def gradcam(
    features: Module, classifier: Module, input_image: Tensor, n_top_classes: int = 3
) -> Tensor:
    """Performs GradCAM on an input image.

    For further explanations about GradCam, see https://arxiv.org/abs/1610.02391.

    Args:
        features: the spatial feature part of the model, before classifier
        classifier: the classifier part of the model, after spatial features
        input_image: image tensor of dimensions (c, h, w)
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a GradCAM heatmap of dimensions (h, w)

    """
    # Prepare model and input
    model = Sequential(features, classifier)
    model.to(midnite.get_device()).eval()
    input_image = _prepare_input(input_image)

    out = model(input_image.unsqueeze(0))

    # Get top N classes
    selector = _top_k_selector(out, n_top_classes)

    # Apply GradAM on Classes
    gradam = GradAM(classifier, selector, features, SpatialSplit())
    result = gradam.visualize(input_image.unsqueeze(0))
    return result


def guided_gradcam(
    feature_layers: List[Module],
    classifier_layers: List[Module],
    input_image: Tensor,
    n_top_classes: int = 3,
) -> Tensor:
    """Performs Guided GradCAM on an input image.

    For further explanations about Guided GradCam, see https://arxiv.org/abs/1610.02391.

    Args:
        features: the spatial feature layers of the model, before classifier
        classifier: the classifier layers of the model, after spatial features
        input_image: image tensor of dimensions (c, h, w)
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a Guided GradCAM heatmap of dimensions (h, w)

    """
    # Create scaled up gradcam image
    cam = gradcam(
        Sequential(*feature_layers),
        Sequential(*classifier_layers),
        input_image,
        n_top_classes,
    )
    cam_scaled = _upscale(cam, input_image.size()[1:])

    # Create guided backprop image
    guided_backprop = guided_backpropagation(
        feature_layers + classifier_layers, input_image, n_top_classes
    )
    # Multiply
    return cam_scaled.mul_(guided_backprop)


def class_visualization(model: Module, class_index: int) -> Tensor:
    """Visualizes a class for a classification network.

    Args:
        model: the model to visualize for
        class_index: the index of the class to visualize

    """
    if class_index < 0:
        raise ValueError(f"Invalid class: {class_index}")

    img = PixelActivation(
        model,
        SplitSelector(NeuronSplit(), [class_index]),
        opt_n=500,
        iter_n=20,
        init_size=50,
        transform=RandomTransform(scale_fac=0)
        + BilateralTransform()
        + ResizeTransform(1.1),
        regularization=[TVRegularization(5e1), WeightDecay(1e-9)],
    ).visualize()

    return PixelActivation(
        model,
        SplitSelector(NeuronSplit(), [class_index]),
        opt_n=100,
        iter_n=int(50),
        transform=RandomTransform() + BilateralTransform(),
        regularization=[TVRegularization(), WeightDecay()],
    ).visualize(img)
