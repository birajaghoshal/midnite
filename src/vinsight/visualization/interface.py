"""General interface of the visualization building blocks"""
from abc import ABC
from abc import abstractmethod
from itertools import product
from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module


class LayerSplit(ABC):
    """Abstract base class for splits of a layer.

    A 'split' represents a way to 'slice the cube' of the spatial and
    channel positions of a layer. For more information, see
    https://distill.pub/2018/building-blocks/

    """

    @abstractmethod
    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        """Returns a split, i.e. all masks.

        Args:
            size: the size of the layer to split as (c, h, w) tuple

        Returns:
            a list of masks containing a mask for each split

        """
        raise NotImplementedError()

    @abstractmethod
    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        """Returns a single mask for the indexed split.

        Args:
            index: the index of the mask to get from the split
            size: the size of the layer to split as (c, h, w) tuple

        Returns:
            a single split mask

        """
        raise NotImplementedError()

    def get_mean(self, input_):
        """Returns the mean of an input along the split dimension.

        Args:
            input_: Tensor for which the mean should be computed
        Returns:
            a tensor of mean values.

        """
        raise NotImplementedError()


class NeuronSplit(LayerSplit):
    """Split a layer neuron-wise, i.e. per single value."""

    def get_split(self, size: List[int]) -> List[Tensor]:
        indexes = product(*map(range, size))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: List[int]) -> Tensor:
        mask = torch.zeros(tuple(size))
        mask[tuple(index)] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            raise ValueError("Channel index needs one dimension. Got: ", input_.size())
        return input_


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

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            raise ValueError("Channel index needs one dimension. Got: ", input_.size())
        return input_.mean(-1).mean(-1)


class ChannelSplit(LayerSplit):
    """Split a layer by its channels."""

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = map(lambda idx: [idx], range(size[0]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 1:
            raise ValueError("Channel index needs one dimension. Got: ", index)
        mask = torch.zeros(*size)
        mask[index[0]] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            raise ValueError("Channel index needs one dimension. Got: ", input_.size())
        return input_.mean(0)


# TODO group split with matrix factorization
# class GroupSplit(LayerSplit):


class NeuronSelector:
    """Selects a number of neurons according to a specific split"""

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

    def __init__(
        self,
        inspection_layers: List[Module],
        top_layer_selector: NeuronSelector,
        base_layers: List[Module],
        bottom_layer_split: LayerSplit,
    ):
        """
        Args:
            inspection_layers: the list of ajacent layers to execute the method on
            top_layer_selector: the target split for analyzing attribution
            base_layers: list of previous layers up to the bottom layer

        """
        if len(inspection_layers) == 0:
            raise ValueError("Must specify at least one top layer")
        self.inspection_layers = inspection_layers
        self.top_layer_selector = top_layer_selector
        self.base_layers = base_layers
        self.bottom_layer_split = bottom_layer_split

    @abstractmethod
    def visualize(self, input_: Tensor) -> Tensor:
        """Abstract method to call attribution method

        Args:
            input_: the input tensor

        Returns: a tensor showing attribution

        """
        raise NotImplementedError()


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
        if len(layers) == 0:
            raise ValueError("Must specify at least one layer")
        self.layers = layers
        self.top_layer_selector = top_layer_selector

    @abstractmethod
    def visualize(self, input_: Optional[Tensor] = None) -> Tensor:
        """Visualizes an activation.

        Args:
            input_: an optional input image, usually a prior. Of shape (height, width,
             channels).

        Returns:
            an activation visualization of shape (h, w, c).

        """
        raise NotImplementedError()
