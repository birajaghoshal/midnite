"""Concrete implementations of occlusion methods."""
from typing import List

import torch
import tqdm
from torch import Tensor
from torch.nn import functional
from torch.nn import Module

import midnite
from ...base.interface import Attribution
from ...base.interface import LayerSplit
from ...base.interface import NeuronSelector


class Occlusion(Attribution):
    """Calculates attribution by occluding parts of the image and comparing outputs"""

    def __init__(
        self,
        net: Module,
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
        chunk_size: List[int],
        stride: List[int],
        norm=2,
    ):
        """
        Args:
            net: part of the network to analyze
            top_layer_selector: specifies elected neurons in the output layer
            bottom_layer_split: neuron split of the bottom layer (i.e., input image)
            chunk_size: chunk size in number of strides, for each dimension
            stride: number of elements (usually pixels) to go through per step,
             for each dimension
            norm: the norm to be computed
        """
        super().__init__([], net, top_layer_selector, bottom_layer_split)
        if not len(chunk_size) == len(stride):
            raise ValueError("Stride and chunk size must have same dimensionality")
        self.chunk_size = chunk_size
        self.stride = stride
        if norm < 1:
            raise ValueError("Must be valid distance norm")
        self.norm = norm

    @classmethod
    def _smear_matrix(cls, size: int, smear: int) -> Tensor:
        """Creates a smear matrix with given size and smear factor."""
        # Identity
        matrix = torch.eye(size, device=midnite.get_device())
        # Add smear diagonals
        for i in range(1, smear):
            matrix[i:, :-i] += torch.eye(size - i, device=midnite.get_device())
        return matrix.clamp_(max=1)

    def _chunk_matrixes(self, input_size: List[int]) -> List[Tensor]:
        """For every dimension, builds the chunk matrix of chunk + input size."""
        if not len(input_size) == len(self.chunk_size):
            raise ValueError("Chunks size must be given for every input dimension")
        return list(
            map(
                lambda sizes: self._smear_matrix(sizes[0] + sizes[1], sizes[1]),
                zip(input_size, self.chunk_size),
            )
        )

    def _chunk_mask(self, pixel_mask: Tensor, chunk_matrixes: List[Tensor]) -> Tensor:
        """Translates a pixel selector mask into a chunk selector mask."""
        sizes = list(zip(pixel_mask.size(), self.chunk_size))
        mask_size = tuple(map(lambda s: s[0] + s[1], sizes))
        # Create mask with initial pixel mask
        mask = torch.zeros(mask_size, device=midnite.get_device())
        mask[tuple(map(lambda s: slice(None, s), pixel_mask.size()))] = pixel_mask
        # build mask
        for i, matrix in enumerate(chunk_matrixes):
            mask = (matrix @ mask.transpose(1, i)).transpose(1, i)
        # Get relevant mask area
        mask_area = tuple(
            map(lambda s: slice((s[1] - 1) // 2, s[0] + ((s[1] - 1) // 2)), sizes)
        )
        # Normalize indexes that were selected multiple times
        return mask[mask_area].clamp_(max=1)

    @classmethod
    def _remove_chunk(cls, input_: Tensor, chunk_mask: Tensor) -> Tensor:
        # Upsample to input image dimensions (use volumetric to get 3d upsample)
        mask = functional.interpolate(
            chunk_mask.unsqueeze(1), tuple(input_.squeeze(0).size())
        ).squeeze(1)
        # Fill chunk with grey
        return input_.mul_(1 - mask).add_(mask * 0.5)

    def visualize(self, input_: Tensor) -> Tensor:
        # Preparation
        input_ = input_.clone().detach().to(midnite.get_device())
        self.black_box_net.to(midnite.get_device())

        # Make reference prediction
        pred = self.black_box_net(input_).squeeze(0)
        # Store top layer mask for the future
        pred_mask = self.top_layer_selector.get_mask(list(pred.size()))
        pred_mask.div_(pred_mask.sum())
        pred.mul_(pred_mask)

        # Dimensions of subsampled image
        subsample_dims = list(
            map(lambda t: t[0] // t[1], zip(input_.squeeze(0).size(), self.stride))
        )

        # Calculate chunk matrixes once and store
        chunk_matrixes = self._chunk_matrixes(subsample_dims)

        # For every pixel in subsampled image, measure difference to true prediction
        result_img = torch.zeros(*subsample_dims, device=midnite.get_device())
        for pixel in tqdm.tqdm(self.bottom_layer_split.get_split(subsample_dims)):
            mask = self._chunk_mask(pixel, chunk_matrixes)
            img = self._remove_chunk(input_.clone(), mask.unsqueeze(0))
            out = self.black_box_net(img).squeeze(0).mul_(pred_mask)
            result_img.add_(mask.squeeze(0).mul_(out.dist(pred, self.norm).item()))

        return self.bottom_layer_split.get_mean(result_img)
