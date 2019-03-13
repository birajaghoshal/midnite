"""Some utils for loading and normalizing data."""
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torchvision.transforms import Normalize


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


def alexnet_transform():
    """Sets the image transformations for AlexNet.
     Necessary dimensions: ( 1 x 3 x 227 x 227 )

    Returns: Transform object for the data

    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform


def get_example_from_path(path_to_img: str) -> Tensor:
    """Retrieves and converts an image to a processable torch tensor for AlexNet

    Args:
        path_to_img: specify the path of the image to be retrieved

    Returns: an example image for AlexNet

    """
    abs_path_to_img = Path(path_to_img).resolve()
    transform = alexnet_transform()

    with open(abs_path_to_img, "rb") as f:
        img = Image.open(f)
        img.convert("RGB")
        img = transform(img)

    return torch.unsqueeze(img, dim=0)


def get_random_example() -> Tensor:
    """Creates a random image.

    Returns: a tensor of size (1, 3, 227, 227) that represents a random image

    """
    random_img = torch.div(
        torch.randint(low=0, high=255, size=(3, 227, 227)).float(), 255.0
    )
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.unsqueeze(normalize(random_img), dim=0)
