import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torchvision.transforms import Normalize


def load_imagenet_dataset(path_to_imagenet: str, transform, batch_size: int) -> list:
    """ Use dataloader to handle image data from image folder "datasets"

    Args:
        path_to_imagenet: path of the imagenet TEST dataset
        batch_size: number of examples that should be retrieved
        transform: torch transform object specifying the image transformation for used architecture

    Returns:
        input_batch : batch with processed images, which can be used for prediction by AlexNet

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
    """Sets the image transformations for AlexNet. Necessary dimensions: ( 1 x 3 x 227 x 227 )

    Returns:
         transform: Transform object for the data
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return transform


def get_example_from_path(abspath_to_img: str) -> Tensor:
    """Retrieves and converts an image to a processable torch tensor for AlexNet

    Args:
        abspath_to_img: specify the absolute path of the image to be retrieved

    Returns:
        input_img:  an example image for AlexNet
    """
    transform = alexnet_transform()

    with open(abspath_to_img, "rb") as f:
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
