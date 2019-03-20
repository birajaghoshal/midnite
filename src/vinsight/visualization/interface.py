"""General interface of the visualization building blocks"""
from abc import ABC
from abc import abstractmethod
from itertools import product
from typing import List
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module


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


class Attribution(ABC):
    """Abstract base class for attribution methods.

    Attribution methods aim to extract insights about the effect that neurons in
    different layers have on each other.

    """

    def __init__(self, layers: List[Module], bottom_layer_split: LayerSplit):
        """
        Args:
            layers: the list of adjacent layers to execute the method on
            bottom_layer_split: the split starting from which information are propagated
             through the network

        """
        self.layers = layers
        self.bottom_layer_split = bottom_layer_split

    @abstractmethod
    def visualize(self, input_: Tensor) -> Tensor:
        """Abstract method to call attribution method

        Args:
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
