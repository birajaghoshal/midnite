"""Some utils for loading and normalizing data."""
from enum import auto
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torchvision.transforms import Normalize


class DataConfig(Enum):
    """Available data configs for networks."""

    ALEX_NET = (auto(),)
    FCN32 = (auto(),)


fcn_mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
alexnet_mean = np.array([0.485, 0.456, 0.406])
alexnet_var = np.array([0.229, 0.224, 0.225])


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

    img = Image.open(abs_path_to_img)
    # Workaround to check default tensor device as there is no method for it
    device = torch.device("cuda" if torch.tensor([]).is_cuda else "cpu")

    if config is DataConfig.ALEX_NET:
        img.convert("RGB")
        img = transforms.Compose([transforms.Resize(227), transforms.CenterCrop(227)])(
            img
        )
        img = (img - 255 * alexnet_mean) / (255 * alexnet_var)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
    elif config is DataConfig.FCN32:
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= fcn_mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
    else:
        raise ValueError("Invalid config.")

    return torch.unsqueeze(img, dim=0).float().requires_grad_(False).to(device)


def get_random_example(config: DataConfig) -> Tensor:
    """Creates a random image.

    Returns: a tensor of size (1, 3, 227, 227) that represents a random image

    """
    # Workaround to check default tensor device as there is no method for it
    device = torch.device("cuda" if torch.tensor([]).is_cuda else "cpu")

    if config is DataConfig.ALEX_NET:
        random_img = torch.div(
            torch.randint(low=0, high=255, size=(3, 227, 227)).float(), 255.0
        )
        random_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            random_img
        )
    elif config is DataConfig.FCN32:
        random_img = np.random.randint(low=0, high=255, size=(3, 640, 488), dtype=float)
        random_img -= fcn_mean_bgr
        random_img = torch.from_numpy(random_img)
    else:
        raise ValueError("Invalid config.")

    return torch.unsqueeze(random_img, dim=0).float().requires_grad_(False).to(device)
