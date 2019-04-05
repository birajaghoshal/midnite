import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.modules import Sequential

from midnite import get_device
from midnite.visualization import base
from midnite.visualization.base import SimpleSelector


def gradcam(
    features: Module,
    classifier: Module,
    input_image: Tensor,
    overlay: bool = True,
    n_top_classes: int = 3,
):
    """Performs GradAM on an input image, as described in #TODO.

    Args:
        model: the model to perform gradcam on
        input_image: image tensor of dimensions (c, h, w)
        overlay: whether to upsample the result and overlay over the input image
        n_top_classes: the number of classes to calculate GradCAM for

    Returns:
        a GradAM image
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
    gradcam = base.GradAM(
        classifier, SimpleSelector(classes_mask), features, base.SpatialSplit()
    )
    result = gradcam.visualize(input_image)

    if not overlay:
        return result
    else:
        # TODO upsample
        return result
