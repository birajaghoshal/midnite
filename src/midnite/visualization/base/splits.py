"""General interface of the visualization building blocks"""
import itertools
import logging
from typing import List

import torch
from torch import Tensor

import midnite
from ..base.interface import LayerSplit
from ..base.interface import NeuronSelector

log = logging.getLogger(__name__)


class Identity(LayerSplit):
    """Identity selector that implements LayerSplit interface."""

    def fill_dimensions(self, input_: Tensor, num_dimensions: int = 3) -> Tensor:
        if not len(input_.size()) == 0:
            raise ValueError(
                f"Cannot specify element for identity. Got: {input_.size()}"
            )
        return super().fill_dimensions(input_, num_dimensions)

    def invert(self) -> LayerSplit:
        return NeuronSplit()

    def get_split(self, size: List[int]) -> List[Tensor]:
        return [self.get_mask([], size)]

    def get_mask(self, index: List[int], size: List[int]) -> Tensor:
        if len(index) > 0:
            raise ValueError("No index required for identity split")
        return torch.ones(tuple(size), device=midnite.get_device())

    def get_mean(self, input_):
        return input_.mean()


class NeuronSplit(LayerSplit):
    """Split a layer neuron-wise, i.e. per single value."""

    def invert(self) -> LayerSplit:
        return Identity()

    def get_split(self, size: List[int]) -> List[Tensor]:
        indexes = itertools.product(*map(range, size))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: List[int]) -> Tensor:
        mask = torch.zeros(tuple(size), device=midnite.get_device())
        mask[tuple(index)] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            log.warning(f"Input normally has 3 dimensions. Got: {input_.size()}")
        return input_


class SpatialSplit(LayerSplit):
    """Split a layer by spatial positions."""

    def fill_dimensions(self, input_: Tensor, num_dimensions: int = 3):
        if not len(input_.size()) == 2:
            log.warning(f"Input should have 2 spatial dimensions. Got: {input_.size()}")
        if num_dimensions == 2 and len(input_.size()) == 2:
            return input_
        else:
            return super().fill_dimensions(input_.unsqueeze(0), num_dimensions)

    def invert(self) -> LayerSplit:
        return ChannelSplit()

    def get_split(self, size: List[int]) -> List[Tensor]:
        indexes = itertools.product(*map(range, size[1:]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: List[int]) -> Tensor:
        if not len(index) == 2:
            raise ValueError("Spatial index need two dimensions. Got: ", index)
        mask = torch.zeros(*size, device=midnite.get_device())
        mask[:, index[0], index[1]] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            log.warning(f"Input normally has 3 dimensions. Got: {input_.size()}")
        # mean over channel dimension, so that there is one mean value for each spatial
        return input_.mean(0)


class ChannelSplit(LayerSplit):
    """Split a layer by its channels."""

    def fill_dimensions(self, input_: Tensor, num_dimensions: int = 3) -> Tensor:
        if not len(input_.size()) == 1:
            log.warning(f"Input should have 1 channel dimensions. Got: {input_.size()}")
        return super().fill_dimensions(input_, num_dimensions)

    def invert(self) -> LayerSplit:
        return SpatialSplit()

    def get_split(self, size: List[int]) -> List[Tensor]:
        indexes = map(lambda idx: [idx], range(size[0]))
        return list(map(lambda idx: self.get_mask(idx, size), indexes))

    def get_mask(self, index: List[int], size: List[int]) -> Tensor:
        if not len(index) == 1:
            raise ValueError(f"Channel index needs one dimension. Got: {index}")
        mask = torch.zeros(*size, device=midnite.get_device())
        mask[index[0]] = 1
        return mask

    def get_mean(self, input_: Tensor) -> Tensor:
        if not len(input_.size()) == 3:
            log.warning(f"Input normally has 3 dimensions. Got: {input_.size()}")
        # mean over spatials, s.t. there is one mean value for each channel
        return input_.mean(-1).mean(-1)


class GroupSplit(LayerSplit):
    """Splits a layer by a decomposition, usually factorized by (spatials, channels)."""

    def __init__(self, left: Tensor, right: Tensor):
        """

        Args:
            left: the left decomposition matrix. Of shape (..., N) where N is the number
             of groups in the decomposition
            right: the right decomposition matrix. Of shape (N, ...)

        """
        if not left.size(-1) == right.size(0):
            raise ValueError(
                f"Not a valid decomposition. Group dimensions must be exqual. Got: "
                f" {left.size(-1)}, {right.size(0)}"
            )
        self.num_groups = left.size(-1)
        self.left = left
        self.right = right

    def invert(self):
        # TODO complex group split
        raise NotImplementedError()

    def get_split(self, size: List[int]) -> List[Tensor]:
        indexes = range(self.num_groups)
        return list(map(lambda i: self.get_mask([i], size), indexes))

    def get_mask(self, index: List[int], _: List[int]) -> Tensor:
        # TODO interpolate result mask to size dimensions
        if not len(index) == 1:
            raise ValueError("Need to provide exactly one index for a group")
        idx = index[0]
        return self.left.select(-1, idx) * self.right.select(0, idx)

    def fill_dimensions(self, input_: Tensor, num_dimensions: int = 3) -> Tensor:
        size = tuple(self.left.size()[:-1] + self.right.size()[1:])
        if not size == input_.size():
            raise ValueError(
                f"Input must of incorrect size. Expected: {size},"
                f" got: {tuple(input_.size())}"
            )
        return super().fill_dimensions(input_, num_dimensions)


class SimpleSelector(NeuronSelector):
    """Simply selects neurons based on a pre-defined mask."""

    def __init__(self, mask: Tensor):
        """

        Args:
            mask: the pre-defined mask for this selector

        """
        self.mask = mask

    def get_mask(self, size: List[int]):
        mask_size = list(self.mask.size())
        if not mask_size == size:
            raise ValueError(f"Incorrect size. Expected: {mask_size}, got: {size}")
        return self.mask.to(midnite.get_device())


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

    def get_mask(self, size: List[int]) -> Tensor:
        return self.layer_split.get_mask(self.element, size)
