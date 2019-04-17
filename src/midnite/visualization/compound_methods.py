"""Compound methods created from building block, implementing common use cases."""
from typing import List
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional
from torch.nn import Softmax
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
from midnite.uncertainty import EnsembleLayer
from midnite.uncertainty import MutualInformation
from midnite.uncertainty import PredictionEnsemble
from midnite.visualization.base import Identity
from midnite.visualization.base import Occlusion


def _prepare_input(img: Tensor) -> Tensor:
    img = img.clone()
    if len(img.size()) == 3:
        img = img.unsqueeze(0)
    if not len(img.size()) == 4 or not img.size(1) == 3:
        raise ValueError(f"Not an image. Size: {img.size()}")
    return img.detach().to(midnite.get_device())


def _top_k_mask(out, k):
    mask = torch.zeros_like(out, device=midnite.get_device())
    classes = out.topk(dim=0, k=k)[1]
    for i in range(k):
        mask[classes[i]] = 1
    return mask


def _top_k_selector(net: Module, img: Tensor, k) -> NeuronSelector:
    """Creates a top-k classes selector.

    Args:
        net: network
        img: prepared input image
        k: number of classes

    Returns:
        neuron selector for the top k classes

    """
    net = net.to(midnite.get_device()).eval()
    out = net(img).squeeze(0)
    mask = _top_k_mask(out, k)

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
        input_image: image tensor of dimensions (c, h, w) or (1, c, h, w)
        n_top_classes: the number of classes to take into account

    Returns:
        a saliency heatmap of dimensions (h, w)

    """
    # Find classes to analyze
    input_image = _prepare_input(input_image)
    class_selector = _top_k_selector(Sequential(*layers), input_image, n_top_classes)

    # Apply guided backpropagation
    backprop = GuidedBackpropagation(layers, class_selector, SpatialSplit())
    return backprop.visualize(input_image)


def gradcam(
    features: Module, classifier: Module, input_image: Tensor, n_top_classes: int = 3
) -> Tensor:
    """Performs GradCAM on an input image.

    For further explanations about GradCam, see https://arxiv.org/abs/1610.02391.

    Args:
        features: the spatial feature part of the model, before classifier
        classifier: the classifier part of the model, after spatial features
        input_image: image tensor of dimensions (c, h, w) or (1, c, h, w)
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a GradCAM heatmap of dimensions (h, w)

    """
    # Get selector for top k classes
    input_image = _prepare_input(input_image)
    class_selector = _top_k_selector(
        Sequential(features, classifier), input_image, n_top_classes
    )
    # Apply spatial GradAM on classes
    gradam = GradAM(classifier, class_selector, features, SpatialSplit())
    result = gradam.visualize(input_image)
    return _upscale(result, tuple(input_image.size()[2:]))


def graduam(
    features: Module, classifier: Module, input_image: Tensor, num_samples: int = 50
) -> Tensor:
    """TODO"""
    input_image = _prepare_input(input_image)

    feature_ensemble = PredictionEnsemble(features, num_samples)
    classifier_ensemble = EnsembleLayer(Sequential(classifier, Softmax(dim=1)))

    gradam = GradAM(
        Sequential(classifier_ensemble, MutualInformation()),
        SplitSelector(Identity(), []),
        feature_ensemble,
        SpatialSplit(),
    )
    result = gradam.visualize(input_image).sum(2)
    return _upscale(result, tuple(input_image.size()[2:]))


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
        input_image: image tensor of dimensions (c, h, w) or (1, c, h, w)
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a Guided GradCAM heatmap of dimensions (h, w)

    """
    input_image = _prepare_input(input_image)
    # Create scaled up gradcam image
    cam = gradcam(
        Sequential(*feature_layers),
        Sequential(*classifier_layers),
        input_image,
        n_top_classes,
    )

    # Create guided backprop image
    guided_backprop = guided_backpropagation(
        feature_layers + classifier_layers, input_image, n_top_classes
    )
    # Multiply
    return cam.mul_(guided_backprop)


def occlusion(net: Module, input_image: Tensor, n_top_classes: int = 3) -> Tensor:
    """Creates a attribution heatmap by occluding parts of the input image.

    Args:
        net: the network to visualize attribution for
        input_image: image tensor of dimensions (c, h, w) or (1, c, h, w)
        n_top_classes: the number of classes to account for

    Returns:
        a occlusion heatmap of dimensions (h, w)

    """
    input_image = _prepare_input(input_image)
    class_selector = _top_k_selector(net, input_image, n_top_classes)
    # Apply occlusion
    occlusion_ = Occlusion(net, class_selector, SpatialSplit(), [1, 10, 10], [1, 5, 5])
    result = occlusion_.visualize(input_image)
    return _upscale(result, tuple(input_image.size()[2:]))


def class_visualization(net: Module, class_index: int) -> Tensor:
    """Visualizes a class for a classification network.

    Args:
        net: the network to visualize for
        class_index: the index of the class to visualize

    """
    if class_index < 0:
        raise ValueError(f"Invalid class: {class_index}")

    img = PixelActivation(
        net,
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
        net,
        SplitSelector(NeuronSplit(), [class_index]),
        opt_n=100,
        iter_n=int(50),
        transform=RandomTransform() + BilateralTransform(),
        regularization=[TVRegularization(), WeightDecay()],
    ).visualize(img)
