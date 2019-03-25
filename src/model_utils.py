from enum import auto
from enum import Enum

from torch.nn import Module
from torchvision import models


class Flatten(Module):
    """One layer module that flattens its input.
    This class is used to flatten the output of a layer split for saliency computation."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ModelConfig(Enum):
    """Available model configs for networks."""

    ALEX_NET = (auto(),)
    RES_NET = (auto(),)


def split_model(model: models, model_config: ModelConfig, layer_idx: int):
    """ Generates a model split at a  selected layer
    Args:
        model: model to be split
        model_config: Enum type of the model (AlexNet, ResNet etc.)
        layer_idx: index of the layer where model should be split
    Return:
        bottom_layer: first part of the model, layers from beginning to the layer of inspection
        top_layer: rest of the model
    """
    if model_config == ModelConfig.ALEX_NET:
        if layer_idx not in range(13):
            raise ValueError("Not a valid layer selection AlexNet.")
        else:
            bottom_layers = list(model.features.children())[:layer_idx]
            top_layers = (
                list(model.features.children())[layer_idx:-1]
                + list(model.features.children())[-1:]
                + list(model.avgpool.children())[:]
                + [Flatten()]
                + list(model.classifier.children())[:]
            )
            return bottom_layers, top_layers

    elif model_config == ModelConfig.RES_NET:
        if layer_idx not in range(9):
            raise ValueError("Not a valid layer selection for ResNet.")
        bottom_layers = list(model.children())[:layer_idx]
        top_layers = (
            list(model.children())[layer_idx:-1]
            + [Flatten()]
            + list(model.children())[-1:]
        )
        return bottom_layers, top_layers

    else:
        raise ValueError("Invalid model for this example.")
