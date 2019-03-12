import torch
from torchvision.transforms import Normalize


def load_dataset():
    # TODO
    return None


def get_imagenet_example():
    """ Retreives an example image from imagenet dataset.

    Return: torch tensor with dimensions: (1 x 3 x 227 x 227 )
    """
    # TODO
    return None


def get_random_example():
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    noise_image = normalize(torch.randn(3, 227, 227))
    return torch.unsqueeze(noise_image, dim=0)


def get_ood_example():
    # TODO
    return None
