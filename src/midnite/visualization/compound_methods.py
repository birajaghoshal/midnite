import torch
from torch import Tensor
from torch.nn.functional import interpolate
from torch.nn.modules import Module
from torch.nn.modules import Sequential

from midnite import get_device
from midnite.visualization.base import GradAM
from midnite.visualization.base import SimpleSelector
from midnite.visualization.base import SpatialSplit


def gradcam(
    features: Module,
    classifier: Module,
    input_image: Tensor,
    rescale: bool = True,
    n_top_classes: int = 3,
):
    """Performs GradCAM on an input image.

    For further explanations about GradCam, see https://arxiv.org/abs/1610.02391.

    Args:
        model: the model to perform gradcam on
        input_image: image tensor of dimensions (c, h, w)
        rescale: whether to upsample the result to input dimensions
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a GradCAM heatmap of dimensions (h, w)

    """
    # Prepare model and input
    model = Sequential(features, classifier)
    model.to(get_device()).eval()
    img = input_image.clone().to(get_device()).unsqueeze(dim=0)

    out = model(img).squeeze(dim=0)

    # Get top N classes
    classes_mask = torch.zeros_like(out, device=get_device())
    classes = out.topk(dim=0, k=n_top_classes)[1]
    for i in range(n_top_classes):
        classes_mask[classes[i]] = 1

    # Apply GradAM on Classes
    gradcam = GradAM(classifier, SimpleSelector(classes_mask), features, SpatialSplit())
    result = gradcam.visualize(img)

    if not rescale:
        return result
    else:
        return (
            interpolate(
                result.unsqueeze(0).unsqueeze(0),
                size=tuple(input_image.size()[1:]),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
