"""Custom modules for MC dropout ensembles and uncertainty."""
import logging
from abc import ABC
from abc import abstractmethod
from typing import Tuple
from typing import Union

import torch
import tqdm
from torch import Tensor
from torch.nn import AlphaDropout
from torch.nn import Dropout
from torch.nn import Dropout2d
from torch.nn import Dropout3d
from torch.nn import FeatureAlphaDropout
from torch.nn import Module
from torch.nn import Sequential
from tqdm import trange

import midnite
from . import functional as func

log = logging.getLogger(__name__)


def _is_dropout(layer: Module) -> bool:
    return (
        isinstance(layer, Dropout)
        or isinstance(layer, Dropout2d)
        or isinstance(layer, Dropout3d)
        or isinstance(layer, AlphaDropout)
        or isinstance(layer, FeatureAlphaDropout)
    )


class InnerForwardMixin:
    """Mixin for simple forward passes."""

    def forward(self, input_):
        input_.to(midnite.get_device())
        for module in self.children():
            input_ = module(input_)
        return input_


class StochasticModule(Module, ABC):
    """Interface for stochastic models.

    Stochastic models can make different predictions for the same input, e.g. using an
    active dropout layer at prediction time.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stochastic = False

    def train(self, mode=True):
        super().train(mode)
        self.stochastic = False

        # All recursive submodules, including self
        for module in self.modules():
            if module is not self and isinstance(module, StochasticModule):
                module.stochastic = False
        return self

    def stochastic_eval(self):
        """Sets the model to stochastic evaluation mode.

        Override this to set your module to stochastic mode.

        Returns: self for fluent interface

        """
        if self.training:
            self.eval()

        self.stochastic = True

        # All recursive submodules, including self
        for module in self.modules():
            if module is not self and isinstance(module, StochasticModule):
                module.stochastic_eval()
        return self


class StochasticDropouts(InnerForwardMixin, StochasticModule):
    """Modifies all dropout layers to be in train mode durching stochastic_eval time."""

    def __init__(self, inner: Module):
        """
        Args:
            inner: inner module

        """
        super().__init__()
        self.add_module("inner", inner)

    def stochastic_eval(self):
        super().stochastic_eval()

        for module in self.modules():
            if module is not self and _is_dropout(module):
                module.training = True

        return self


class EnsembleBegin(StochasticModule):
    def __init__(self, num_passes: int = 20):
        """
        Args:
            num_passes: number of stochastic passes in the ensemble

        """
        super().__init__()
        if num_passes < 2:
            raise ValueError("At least two samples are necessary")

        if num_passes < 20:
            log.warning("Using low number of samples may give greatly skewed results")

        self.num_passes = num_passes

    def forward(self, input_: Tensor) -> Tensor:
        input_.to(midnite.get_device())

        # In eval mode, stack ensemble predictions
        if self.stochastic:
            if input_.requires_grad:
                log.warning(
                    "Gradients enabled for ensemble predictions requires large "
                    "amounts of memory"
                )

            return torch.stack((input_,) * self.num_passes, len(input_.size()))
        else:
            return input_


class EnsembleLayer(InnerForwardMixin, StochasticModule):
    """Module for a single layer in ensemble mode.

    May be used between EnsembleBegin and an acquisition function, e.g. PredictiveEntropy.

    """

    def __init__(self, inner: Union[Module, StochasticModule]):
        """
        Args:
            inner: stochastic inner module

        """
        super().__init__()
        self.add_module("inner", inner)

    def forward(self, input_: Tensor) -> Tensor:
        if self.stochastic:
            input_.to(midnite.get_device())
            outs = []
            for i in trange(input_.size(-1)):
                single_input = input_.select(-1, i).clone()
                outs.append(super().forward(single_input))
            return torch.stack(tuple(outs), len(outs[0].size()))
        else:
            return super().forward(input_)


class MeanEnsemble(InnerForwardMixin, StochasticModule):
    """Module for models that want the mean of their ensemble predictions."""

    def __init__(self, inner: Union[Module, StochasticModule], num_passes: int = 20):
        """
        Args:
            inner: stochastic inner module
            num_passes: number of stochastic passes in the ensemble

        """
        super().__init__()
        self.add_module("inner", inner)

        if num_passes < 2:
            raise ValueError("At least two samples are necessary")

        if num_passes < 20:
            log.warning("Using low number of samples may give greatly skewed results")

        self.num_passes = num_passes

    def forward(self, input_: Tensor) -> Tensor:
        if self.stochastic:
            with tqdm.trange(
                self.num_passes - 1, total=self.num_passes
            ) as sample_range:
                pred = super().forward(input_)
                sample_range.update()

                # Calc remaining forward pass
                if input_.requires_grad:
                    log.warning(
                        "Gradients enabled for ensemble predictions requires large "
                        "amounts of memeory"
                    )

                    for _ in sample_range:
                        pred = pred.add(super().forward(input_))
                    return pred.div(self.num_passes)
                else:
                    for _ in sample_range:
                        pred.add_(super().forward(input_))
                    return pred.div_(self.num_passes)
        else:
            return super().forward(input_)


class Acquisition(Module, ABC):
    """Abstract base class for acquisition functions."""

    def __init__(self, per_class=False):
        """
        Args:
            per_class: signals whether the uncertainty measure should be outputted
             in total or per class

        """
        super().__init__()
        self.per_class = per_class

    @abstractmethod
    def measure_uncertainty(self, input_: Tensor) -> Tensor:
        """Abstract method to calculate uncertainty

        Args:
            input_: the input tensor

        Returns:
            an uncertainty tensor

        """
        raise NotImplementedError

    def forward(self, input_: Tensor) -> Tensor:
        input_.to(midnite.get_device())
        uncertainty = self.measure_uncertainty(input_)

        if self.per_class:
            return uncertainty
        else:
            return uncertainty.sum(dim=1)


class PredictiveEntropy(Acquisition):
    """Module to calculate the predictive entropy from an ensemble model.

    Predictive entropy (also called max entropy) measures total uncertainty.

    """

    def measure_uncertainty(self, input_: Tensor) -> Tensor:
        return func.predictive_entropy(input_, inplace=not input_.requires_grad)


class MutualInformationUncertainty(Acquisition):
    """Module to calculate an uncertainty based on mutual information.

    Measures epistemic uncertinty. Also called Bayesian Active Learning by Disagreement
     (BALD).

    """

    def measure_uncertainty(self, input_: Tensor) -> Tensor:
        return func.mutual_information_uncertainty(
            input_, inplace=not input_.requires_grad
        )


class VariationRatio(Acquisition):
    """Module to calculate the variation ratio from an ensemble model.

    Gives a percentage for total predictive uncertainty.

    """

    def measure_uncertainty(self, input_: Tensor) -> Tensor:
        return func.variation_ratio(input_, inplace=not input_.requires_grad)


class PredictionAndUncertainties(Acquisition):
    """Module to conveniently calculate sampled mean and uncertainties."""

    def measure_uncertainty(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        return (
            func.predictive_entropy(input_.clone(), inplace=not input_.requires_grad),
            func.mutual_information_uncertainty(
                input_, inplace=not input_.requires_grad
            ),
        )

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            mean prediction, predictive entropy, mutual information

        """
        input_.to(midnite.get_device())
        pred = input_.mean(dim=(input_.dim() - 1,))
        pe, mi = self.measure_uncertainty(input_)
        if self.per_class:
            return pred, pe, mi
        else:
            return pred, pe.sum(dim=1), mi.sum(dim=1)


class Ensemble(InnerForwardMixin, StochasticModule):
    """Complete ensemble wrapper."""

    def __init__(
        self, begin: EnsembleBegin, acquisition: Acquisition, *layers: EnsembleLayer
    ):
        """
        Args:
            start: the ensemble start
            *layers: the intermediate layers
            acquisition: the final acquisition function
        """
        super().__init__()
        self.add_module("start", begin)
        self.add_module("layers", Sequential(*layers))
        self.add_module("acquisition", acquisition)
