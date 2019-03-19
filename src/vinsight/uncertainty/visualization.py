"""POC for the visualization interface"""
from abc import ABC
from abc import abstractmethod
from typing import List

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential


class LayerSplit(ABC):
    @abstractmethod
    def split(self, layer):
        pass


class NeuronSplit(LayerSplit):
    def split(self, layer):
        # TODO
        pass


class SpatialSplit(LayerSplit):
    def split(self, layer):
        # TODO
        pass


class ChannelSplit(LayerSplit):
    def split(self, layer):
        # TODO
        pass


class GroupSplit(LayerSplit):
    def __init__(self, factorizaiton):
        self.factorization = factorizaiton

    def split(self, layer):
        # TODO
        pass


class NeuronSelector:
    def __init__(self, layer_split: LayerSplit, element: Tensor):
        self.layer_split = layer_split
        self.element = element


class Attribution(ABC):
    def __init__(self, layers: List[Module], bottom_layer_split: LayerSplit):
        self.layers = layers
        self.bottom_layer_split = bottom_layer_split

    @abstractmethod
    def visualize(self, input_: Tensor) -> Tensor:
        pass


class Activation(ABC):
    def __init__(self, layers: List[Module], top_layer_selector: NeuronSelector):
        self.layers = layers
        self.top_layer_selector = top_layer_selector


class SaliencyMap(Attribution):
    def visualize(selfs, input_: Tensor) -> Tensor:
        # TODO
        pass


class CodeInversion(Activation):
    def visualize(self, input_: Tensor) -> Tensor:
        # TODO
        pass


if __name__ == "main":
    model = Sequential(Module(), Module(), Module())
    input_ = torch.zeros(512, 50, 50)
    # Two visualization from https://distill.pub/2018/building-blocks/

    neurons = Attribution

    # 1st/2nd visualization under "What Does the Network See?". Some indexing missing.
    mixed4d = [model.__getitem__(2)]
    max_map = SaliencyMap(mixed4d, SpatialSplit()).visualize(input_).argmax(dim=2)

    for neuron in NeuronSplit().split(max_map):
        neuron = max_map[neuron]
        vis = CodeInversion(mixed4d, NeuronSelector(SpatialSplit(), neuron))
