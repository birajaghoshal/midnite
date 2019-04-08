"""General interface of the visualization building blocks"""
from itertools import product
from typing import List
from typing import Tuple

import torch
from torch import Tensor

import midnite
from midnite.visualization.base.interface import LayerSplit
from midnite.visualization.base.interface import NeuronSelector


class Identity(LayerSplit):
    """Identity selector that implements LayerSplit interface."""

    def fill_dimensions(self, input_):
        if not len(input_.size()) == 0:
            raise ValueError(
                f"Cannot specify element for identity. Got: {input_.size()}"
            )
        return torch.ones((1, 1, 1), device=midnite.get_device())

    def invert(self) -> LayerSplit:
        return NeuronSplit()

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        return [self.get_mask([], size)]

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if len(index) > 0:
            raise ValueError("No index required for identity split")
        return torch.ones(size)

    def get_mean(self, input_):
        return input_.mean()


class NeuronSplit(LayerSplit):
    """Split a layer neuron-wise, i.e. per single value."""

    def fill_dimensions(self, input_):
        if not len(input_.size()) == 3:
            raise ValueError(f"Input must have three dimensions. Got: {input_.size()}")
        return input_

    def invert(self) -> LayerSplit:
        return Identity()

    def get_split(self, size: List[int]) -> List[Tensor]:
        indexes = product(*map(range, size))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: List[int]) -> Tensor:
        mask = torch.zeros(tuple(size), device=midnite.get_device())
        mask[tuple(index)] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        return input_


class SpatialSplit(LayerSplit):
    """Split a layer by spatial positions."""

    def fill_dimensions(self, input_):
        if not len(input_.size()) == 2:
            raise ValueError(
                f"Input needs to have 2 spatial dimensions: (h, w). Got: {input_.size()}"
            )
        return input_.unsqueeze(dim=0)

    def invert(self) -> LayerSplit:
        return ChannelSplit()

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = product(*map(range, size[1:]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 2:
            raise ValueError("Spatial index need two dimensions. Got: ", index)
        mask = torch.zeros(*size, device=midnite.get_device())
        mask[:, index[0], index[1]] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            raise ValueError(
                f"Input needs to have 3 dimensions: (c, h, w). Got: {input_.size()}"
            )
        # mean over channel dimension, so that there is one mean value for each spatial
        return input_.mean(0)


class ChannelSplit(LayerSplit):
    """Split a layer by its channels."""

    def fill_dimensions(self, input_):
        if not len(input_.size()) == 1:
            raise ValueError(
                "Input needs to have 1 channel dimensions: (c,). Got: ",
                len(input_.size()),
            )
        return input_.unsqueeze(dim=1).unsqueeze(dim=2)

    def invert(self) -> LayerSplit:
        return SpatialSplit()

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = map(lambda idx: [idx], range(size[0]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 1:
            raise ValueError(f"Channel index needs one dimension. Got: {index}")
        mask = torch.zeros(*size, device=midnite.get_device())
        mask[index[0]] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        # mean over spatials, s.t. there is one mean value for each channel
        return input_.mean(-1).mean(-1)


class GroupSplit(LayerSplit):
    """Splits a layer by a decomposition, usually factorized by (spatials, channels)."""

    def __init__(self, decomposition: Tuple[Tensor, Tensor]):
        if not decomposition[0].size(-1) == decomposition[1].size(0):
            raise ValueError(
                f"Not a valid decomposition. Group dimensions must be exqual. Got: "
                + f" {decomposition[0].size(-1)}, {decomposition[1].size(0)}"
            )
        self.num_groups = decomposition[0].size(-1)
        self.decomposition = decomposition

    def invert(self):
        return GroupSplit(
            (
                self.decomposition[1].transpose(0, -1),
                self.decomposition[0].transpose(0, -1),
            )
        )

    def get_split(self, size: Tuple[int, int, int]) -> List[Tensor]:
        indexes = range(self.num_groups)
        return list(map(lambda i: self.get_mask([i], size), indexes))

    def get_mask(self, index: List[int], _: Tuple[int, int, int]) -> Tensor:
        if not len(index) == 1:
            raise ValueError("Need to provide exactly one index for a group")
        idx = index[0]
        return self.decomposition[0].select(-1, idx) * self.decomposition[1].select(
            0, idx
        )

    def fill_dimensions(self, input_):
        return input_


class SimpleSelector(NeuronSelector):
    """Simply selects neurons based on a pre-defined mask."""

    def __init__(self, mask: Tensor):
        self.mask = mask

    def get_mask(self, size: Tuple[int, int, int]):
        mask_size = self.mask.size()
        if not tuple(mask_size) == size:
            raise ValueError(f"Incorrect size. Expected: {mask_size}, got: {size}")
        return self.mask


class SplitSelector(NeuronSelector):
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
        return self.layer_split.get_mask(self.element, size)
