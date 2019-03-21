"""General interface of the visualization building blocks"""
from abc import ABC
from abc import abstractmethod
from itertools import product
from typing import List
from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torch.nn import Module


class ModelSplit(ABC):
    """Splits the model into parts."""

    def get_split(self, model: torchvision.models, layer: int) -> List[Module]:
        """Returns a split of a model.

        Args:
            model: the model to be split
            layer: layer index where the model should be split

        Returns:
            split: a list of Sequentials: split 1 goes until selected layer index,
                split 2 until the end of feature layers, lastly, the flattened classifier

        """
        bottom_layers = list(model.children())[:layer]
        top_layers = (
            list(model.children())[layer:-1] + [Flatten()] + list(model.children())[-1:]
        )
        bottom_layer_split = SpatialSplit()

        return top_layers, bottom_layers, bottom_layer_split


class LayerSplit(ABC):
    """Abstract base class for splits of a layer."""

    @abstractmethod
    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        """Returns a split, i.e. all masks.

        Args:
            size: the size of the layer to split as (c, h, w) tuple

        Returns:
            a list of masks containing a mask for each split

        """
        pass

    @abstractmethod
    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        """Returns a single mask for the indexed split.

        Args:
            index: the index of the mask to get from the split
            size: the size of the layer to split as (c, h, w) tuple

        Returns:
            a single split mask

        """
        pass


class NeuronSplit(LayerSplit):
    """Split a layer neuron-wise, i.e. per single value."""

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = product(*map(range, size))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 3:
            raise ValueError("Index has to have 3 dimensions")
        mask = torch.zeros(*size)
        mask[tuple(index)] = 1
        return mask


class SpatialSplit(LayerSplit):
    """Split a layer by spatial positions."""

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = product(*map(range, size[1:]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 2:
            raise ValueError("Spatial index need two dimensions. Got: ", index)
        mask = torch.zeros(*size)
        mask[:, index[0], index[1]] = 1
        return mask


class ChannelSplit(LayerSplit):
    """Split a layer by its channels."""

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = map(lambda idx: [idx], range(size[1]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 1:
            raise ValueError("Channel index needs one dimension. Got: ", index)
        mask = torch.zeros(*size)
        mask[index[0]] = 1
        return mask


# class GroupSplit(LayerSplit):


class NeuronSelector:
    """Select a neurons according to a specific split"""

    def __init__(self, layer_split: LayerSplit, element: List[int]):
        """
        Args:
            layer_split: the split to use
            element: the element in the split

        """
        self.layer_split = layer_split
        self.element = element

    def get_mask(self, size: Tuple[int, int, int]) -> Tensor:
        """Get the mask for the specified neurons

        Args:
            size: size of the layer

        Returns:
            a mask of the selected neurons

        """
        return self.layer_split.get_mask(self.element, size)


class Flatten(Module):
    """One layer module that flattens its input.
    This class is used to flatten the output of a layer split for saliency computation."""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Attribution(ABC):
    """Abstract base class for attribution methods.

    Attribution methods aim to extract insights about the effect that neurons in
    different layers have on each other.

    """

    def __init__(self, top_layers: List[Module], bottom_layer_split: LayerSplit):
        """
        Args:
            top_layers: the list of adjacent layers from ModelSplit
            bottom_layer_split: the split starting from which information are propagated
             through the network

        """
        self.layers = top_layers
        self.bottom_layer_split = bottom_layer_split

    @abstractmethod
    def visualize(self, selected_class: int, input_: Tensor) -> Tensor:
        """Abstract method to call attribution method

        Args:
            selected_class: the class for which the saliency should be computed
            input_: the input tensor

        Returns: a tensor showing attribution

        """
        pass


class Activation(ABC):
    """Abstract base class for activation methods.

    Activation methods aim to visualize some properties of the activation of a layer or
    a number of layers.

    """

    def __init__(self, layers: List[Module], top_layer_selector: NeuronSelector):
        """
        Args:
            layers: the list of adjacent layers to execute the method on
            top_layer_selector: the split on which the activation properties are
             extracted

        """
        self.layers = layers
        self.top_layer_selector = top_layer_selector
