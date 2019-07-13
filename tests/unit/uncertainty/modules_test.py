"""Tests for the custom nn modules"""
import pytest
import torch
from assertpy import assert_that
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from torch.nn import Dropout
from torch.nn import Sequential

from midnite.uncertainty import functional
from midnite.uncertainty.modules import Acquisition
from midnite.uncertainty.modules import EnsembleBegin
from midnite.uncertainty.modules import EnsembleLayer
from midnite.uncertainty.modules import MeanEnsemble
from midnite.uncertainty.modules import MutualInformationUncertainty
from midnite.uncertainty.modules import PredictionAndUncertainties
from midnite.uncertainty.modules import PredictiveEntropy
from midnite.uncertainty.modules import StochasticDropouts
from midnite.uncertainty.modules import VariationRatio

# For consistent tests, seed has to be fixed

torch.manual_seed(123456)


def test_stochastic_dropouts():
    """Checks if dropouts are active in stochastic eval mode."""
    layer = StochasticDropouts(Sequential(Dropout(), Dropout()))

    input_ = torch.ones((1, 40))
    layer.stochastic_eval()
    out = layer(input_).squeeze(0).numpy()

    assert_that(out).contains(0.0)
    assert_that(out).contains(4.0)
    assert_that(out).contains_only(4.0, 0.0)


def test_stochastic_dropouts_eval():
    """Checks that dropouts cam be deactivated again."""
    layer = StochasticDropouts(Sequential(Dropout(), Dropout()))

    input_ = torch.ones((1, 40))
    layer.stochastic_eval()
    layer.eval()
    out = layer(input_).squeeze(0).numpy()

    assert_that(out).contains_only(1.0)


def test_ensemble_begin():
    """Test the ensemble begin layer"""
    # In non-stochastic mode
    begin = EnsembleBegin(25)
    input_ = torch.tensor([[1, 2, 3, 4, 5]])
    assert_array_equal(begin(input_), input_)

    # In stochastic mode
    begin.stochastic_eval()
    out = begin(input_)
    assert_that(out.size()).is_equal_to((1, 5, 25))
    for i in range(25):
        assert_array_equal(out[:, :, i], [[1, 2, 3, 4, 5]])


@pytest.mark.parametrize("requires_grad", [True, False])
def test_mean_ensemble_layer(mocker, requires_grad):
    """Checks if the layer correctly calculates an ensemble mean."""
    inner = StochasticDropouts(Dropout())
    mocker.spy(inner, "forward")
    ensemble = MeanEnsemble(inner, 30)
    ensemble.stochastic_eval()

    input_ = torch.ones((1, 10), requires_grad=requires_grad)
    output = ensemble(input_.clone())

    assert torch.allclose(output, torch.ones((1, 10)), atol=0.4)
    assert not torch.allclose(output, torch.ones((1, 10)))
    assert_that(inner.forward.call_count).is_equal_to(30)

    ensemble.eval()
    assert_array_equal(ensemble(input_.clone()).detach(), input_.detach())


def test_ensemble_layer(mocker):
    """Checks if the ensemble layer does the sampling correctly."""
    inner = StochasticDropouts(Dropout())
    mocker.spy(inner, "forward")
    ensemble = EnsembleLayer(inner)
    ensemble.eval()

    input_ = torch.ones((1, 10))
    output = ensemble(input_)
    assert_that(output.size()).is_equal_to((1, 10))
    assert_array_equal(output.squeeze(0), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Stochastic mode
    ensemble.stochastic_eval()
    input_ = torch.ones((1, 10, 25))
    output = ensemble(input_)

    assert_that(output.size()).is_equal_to((1, 10, 25))
    assert torch.all((output == 0.0) + (output == 2.0))
    assert_that(inner.forward.call_count).is_equal_to(26)


def test_acquisition():
    """Checks if the acquisition base class uses per_class flag correctly."""

    class TestAcquisition(Acquisition):
        def measure_uncertainty(self, input_):
            return input_ * 2

    input_ = torch.ones((1, 3, 2))
    assert_array_equal(TestAcquisition()(input_), [[6, 6]])
    assert_array_equal(TestAcquisition(True)(input_), [[[2, 2], [2, 2], [2, 2]]])


def test_pred_entropy(mocker):
    """Predictive entropy layer test"""
    # Patch out functional
    mocker.patch("midnite.uncertainty.functional.predictive_entropy")

    input_ = torch.ones((1, 20, 2))
    PredictiveEntropy()(input_)

    functional.predictive_entropy.assert_called_once_with(input_, inplace=True)


def test_mutual_information(mocker):
    """Mutual information layer test."""
    mocker.patch("midnite.uncertainty.functional.mutual_information_uncertainty")

    input_ = torch.ones(1, 20, 2)
    MutualInformationUncertainty()(input_)

    functional.mutual_information_uncertainty.assert_called_once_with(
        input_, inplace=True
    )


def test_variation_ratio(mocker):
    """Variation ratio layer test."""
    mocker.patch("midnite.uncertainty.functional.variation_ratio")

    input_ = torch.ones(1, 20, 2)
    VariationRatio()(input_)

    functional.variation_ratio.assert_called_once_with(input_, inplace=True)


def test_prediction_and_uncertainties(mocker):
    """Combined output layer test."""
    mocker.patch("midnite.uncertainty.functional.predictive_entropy")
    mocker.patch("midnite.uncertainty.functional.mutual_information_uncertainty")

    input_ = torch.tensor([[1.0, 0.5, 0.3]])
    input_.clone = mocker.Mock(return_value=input_)
    output, _, _ = PredictionAndUncertainties()(input_)

    assert_array_almost_equal(output, [0.6])

    functional.predictive_entropy.assert_called_once_with(input_, inplace=True)
    functional.mutual_information_uncertainty.assert_called_once_with(
        input_, inplace=True
    )


def test_prediction_and_uncertainties_per_class():
    """Checks if the layer uses per_class flag correctly."""
    input_ = torch.ones((1, 3, 2))
    res = PredictionAndUncertainties()(input_)
    assert_that(res[0].size()).is_equal_to((1, 3))
    assert_that(res[1].size()).is_equal_to((1,))
    assert_that(res[2].size()).is_equal_to((1,))

    res = PredictionAndUncertainties(True)(input_)
    assert_that(res[0].size()).is_equal_to((1, 3))
    assert_that(res[1].size()).is_equal_to((1, 3))
    assert_that(res[2].size()).is_equal_to((1, 3))
