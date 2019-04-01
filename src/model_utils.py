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


def split_model_with_classification(
    model: models, model_config: ModelConfig, layer_idx: int
):
    """ Generates a model split at a  selected layer
    Args:
        model: model to be split
        model_config: Enum type of the model (AlexNet, ResNet etc.)
        layer_idx: index of the layer where model should be split
    Return:
        base_layers: first part of the model, inspection_layers from beginning to the layer of inspection
        top_layer: rest of the model
    """
    if model_config == ModelConfig.ALEX_NET:
        if layer_idx not in range(13):
            raise ValueError("Not a valid layer selection AlexNet.")

        base_layers = list(model.features.children())[:layer_idx]
        inspection_layers = (
            list(model.features.children())[layer_idx:-1]
            + list(model.features.children())[-1:]
            + list(model.avgpool.children())[:]
            + [Flatten()]
            + list(model.classifier.children())[:]
        )
        return base_layers, inspection_layers

    elif model_config == ModelConfig.RES_NET:
        if layer_idx not in range(9):
            raise ValueError("Not a valid layer selection for ResNet.")

        base_layers = list(model.children())[:layer_idx]
        inspection_layers = (
            list(model.children())[layer_idx:-1]
            + [Flatten()]
            + list(model.children())[-1:]
        )
        return base_layers, inspection_layers

    else:
        raise ValueError("Invalid model config.")


def split_model_without_classification(model, model_config, layer_idx, output_idx):
    """ Generates a model split at a  selected layer
    Args:
        model: model to be split
        model_config: Enum type of the model (AlexNet, ResNet etc.)
        layer_idx: index of the layer where model should be split
        output_idx: index of the output layer
    Return:
        base_layers: list of inspection_layers from first layer of the model to the selected layer of inspection
        inspection_layers: list of inspection_layers from the selected layer at layer_idx up to the output layer at output_idx
    """
    if model_config == ModelConfig.ALEX_NET:
        if layer_idx not in range(13) or output_idx not in range(13):
            raise ValueError("Not a valid layer selection AlexNet.")

        base_layers = list(model.features.children())[:layer_idx]

        inspection_layers = (
            list(model.features.children())[layer_idx:output_idx]
            + list(model.features.children())[output_idx : output_idx + 1]
        )
        return base_layers, inspection_layers

    else:
        raise ValueError("Invalid model config.")
