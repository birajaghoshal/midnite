# Visualization
The midnite visualization tools can be used on two levels of abstraction:
1. [Specific visualization methods](#specific-visualization-methods)
2. [General building blocks](#building-blocks)

## Specific visualization methods
These methods can be easily used on existing networks.
Some require specific layers or parts of a model (e.g., spatial features and classifier).
For a hands-on, have a look at the [corresponding notebook](notebooks/2_compound_visualization).

Details on how to use all methods can also be found in the [apidocs](api/midnite.visualization.compound_methods).

To this end, the following visualizations are implemented in midnite:
```eval_rst
+------------------------+----------------------------------------------------------------------+------------------------------------+
| name                   | description                                                          | links                              |
+========================+======================================================================+=================+==================+
| guided backpropagation | deconvolution of gradient from higher layers to lower layers         | `notebook <1_>`_|`references <4_>`_|
+------------------------+----------------------------------------------------------------------+                 +------------------+
| (guided) GradCAM       | deconvolution of gradient, weighted by filter importance             |                 |`references <5_>`_|
+------------------------+----------------------------------------------------------------------+-----------------+------------------+
| occlusion              | differences in prediction when occluding parts of the input          |`notebook <2_>`_, `references <6_>`_|
+------------------------+----------------------------------------------------------------------+------------------------------------+
| class visualization    | optimizing input values to maximally incite output layer for a class |`notebook <3_>`_, `references <7_>`_|
+------------------------+----------------------------------------------------------------------+------------------------------------+

.. _1: notebooks/details_gradient_attribution.rst
.. _2: notebooks/details_occlusion.rst
.. _3: notebooks/details_max_mean_activation.rst
.. _4: references.html#guided-backpropagation
.. _5: references.html#gradcam
.. _6: references.html#occlusion
.. _7: references.html#max-mean-activation-optimization
```

## Building blocks
Midnite's lower level visualization methods are implemented as generic _building blocks_.
If you want to use them, there are a few important concepts to understand first.

### Concepts and Interfaces
```eval_rst
All methods are implemented in a consistent building blocks framework, as proposed in :cite:`b-blocks_blocks-interpretability`.
This allows to re-use lots of code and easily add new complex visualization methods.


```

To summarize, neurons of a _specific_ layer can be partitioned in different ways, depending on which information is required:
 - _spatial positions_ can show the pixel-location of relevant information
 - _channels_ often capture specific semantic information
 - _individual neurons_ give very fine-grained information
 - _groups of neurons_ can give a good aggregation view for a specific layer matrix
 
Hence, building blocks do not use specific splits, but use the [LayerSplit](api/midnite.visualization.base.interface.rst#midnite.visualization.base.interface.LayerSplit) interface, whose signature looks like this (implementation and details omitted, see apidoc):

```python
class LayerSplit(ABC):
    @abstractmethod
    def invert(self)
    @abstractmethod
    def get_split(self, size: List[int]) -> List[Tensor]
    @abstractmethod
    def get_mask(self, index: List[int], size: List[int]) -> Tensor
    def get_mean(self, input_) -> Tensor
    def fill_dimensions(self, input_: Tensor, num_dimensions: int = 3) -> Tensor
```

To identify specific neurons, there is a simple [NeuronSelector](api/midnite.visualization.base.interface.rst#midnite.visualization.base.interface.NeuronSelector):
```python
class NeuronSelector(ABC):
    @abstractmethod
    def get_mask(self, size: List[int]) -> Tensor
```

Furthermore, since the building blocks are often interested in parts layers of the network, we use `torch.nn.Module`s or `List[torch.nn.Module]` as abstraction over which part of the network to analyze. 

Finally, visualization methods can be seperated into:
 - _attribution methods_ that try explain the interaction of neurons from different layers and the effect they have on each other
 - _activation methods_ which aim to visualize properties of the activation of specific neurons

Correspondingly, visualization building blocks methods implement the [Attribution](api/midnite.visualization.base.interface.rst#midnite.visualization.base.interface.Attribution) or [Activation](api/midnite.visualization.base.interface.rst#midnite.visualization.base.interface.Activation) interface:
```python
class Attribution(ABC):
    def __init__(
        self,
        white_box_layers: List[Module],
        black_box_net: Optional[Module],
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
    )
    @abstractmethod
    def visualize(self, input_: Tensor) -> Tensor
    
class Activation(ABC):
    def __init__(
        self,
        white_box_layers: List[Module],
        black_box_net: Optional[Module],
        top_layer_selector: NeuronSelector,
    )
    @abstractmethod
    def visualize(self, input_: Optional[Tensor] = None) -> Tensor
```
Since attribution methods are interested in the interaction between layers, they need a split of the bottom layer, which Activation methods generally do not.
On the other hand, activation methods may not need an input image for visualization.


### Implementing a building block
Using those concepts, as an example here we implement backpropagation (which is an [Attribution](api/midnite.visualization.base.interface.rst#midnite.visualization.base.interface.Attribution) method) as simple building block:
```python
class Backpropagation(Attribution):
    def __init__(self,
        net: Module,
        top_layer_selector: NeuronSelector,
        bottom_layer_split: LayerSplit,
    ):
        super().__init__([], net, top_layer_selector, bottom_layer_split)
```

Since we are only interested in the whole network and not specific layers, we call `super` with the whole network and an empty list of layers.

The `visualize` method is then as follows (omitting code about calculating on proper devices, etc.):

```python
    def visualize(self, input_: Tensor) -> Tensor:
        # forward pass
        out = self.black_box_net(input_).squeeze(dim=0)

        # retrieve mean score of top layer selector
        score = (out * self.top_layer_selector.get_mask(list(out.size()))).sum()
        
        # backward pass
        score.backward()

        # mean over bottom layer split dimension
        return self.bottom_layer_split.get_mean(input_.grad.detach().squeeze(dim=0))
```

```eval_rst
.. bibliography:: references.bib
   :labelprefix: B
   :keyprefix: b-
   :style: plain
```