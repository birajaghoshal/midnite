"""Some utils for loading and normalizing data."""
from enum import auto
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor
from torchvision.transforms import Normalize

from vinsight import get_device


class DataConfig(Enum):
    """Available data configs for networks."""

    ALEX_NET = (auto(),)
    FCN32 = (auto(),)


fcn_mean_bgr = torch.tensor([104.00698793, 116.66876762, 122.67891434])
alexnet_mean = torch.tensor([0.485, 0.456, 0.406])
alexnet_var = torch.tensor([0.229, 0.224, 0.225])


def load_imagenet_dataset(path_to_imagenet: str, transform, batch_size: int) -> list:
    """ Use dataloader to handle image data from image folder "data"

    Args:
        path_to_imagenet: path of the imagenet TEST dataset
        batch_size: number of examples that should be retrieved
        transform: torch transform object specifying the image transformation for
         architecture used

    Returns: batch with processed images, which can be used for prediction by AlexNet

    """

    imagenet_data = torchvision.datasets.ImageFolder(
        path_to_imagenet, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        imagenet_data, batch_size=batch_size, shuffle=True
    )

    input_batch = []
    for img, label in data_loader:
        input_batch.append(img)

    return input_batch


def get_example_from_path(path_to_img: str, config: DataConfig) -> Tensor:
    """Retrieves and converts an image to a processable torch tensor for AlexNet

    Args:
        path_to_img: specify the path of the image to be retrieved
        config: The network to retrieve the image for

    Returns: an example image for AlexNet

    """
    abs_path_to_img = Path(path_to_img).resolve()

    with Image.open(abs_path_to_img) as img:
        if config is DataConfig.ALEX_NET:
            img.convert("RGB")
            img = torch.from_numpy(np.array(img)).float().to(get_device())
            img = img.sub_(255 * alexnet_mean).div_(255 * alexnet_var)
        elif config is DataConfig.FCN32:
            # RGB -> BGR
            img = np.array(img)[:, :, ::-1]
            img = torch.from_numpy(np.array(img)).float().to(get_device())
            img.sub_(fcn_mean_bgr)
        else:
            raise ValueError("Invalid config.")

    return torch.unsqueeze(img.permute(2, 0, 1), dim=0).requires_grad_(False)


def get_random_example(config: DataConfig) -> Tensor:
    """Creates a random image.

    Returns: a tensor of size (1, 3, 227, 227) that represents a random image

    """

    if config is DataConfig.ALEX_NET:
        random_img = torch.randint(low=0, high=255, size=(3, 227, 227)).float()
        random_img.to(get_device()).div_(255)
        random_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            random_img
        )
    elif config is DataConfig.FCN32:
        random_img = torch.zeros((3, 640, 488)).to(get_device())
        random_img.add_(torch.randint(low=0, high=255, size=(3, 640, 488)))
        random_img.div_(255).sub_(fcn_mean_bgr)
    else:
        raise ValueError("Invalid config.")

    return torch.unsqueeze(random_img, dim=0).requires_grad_(False)
