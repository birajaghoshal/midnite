"""Tests for the custom nn modules"""
import torch
from assertpy import assert_that
from numpy.testing import assert_array_almost_equal

from midnite.uncertainty import functional
from midnite.uncertainty.modules import MeanEnsemble
from midnite.uncertainty.modules import MutualInformation
from midnite.uncertainty.modules import PredictionAndUncertainties
from midnite.uncertainty.modules import PredictionDropout
from midnite.uncertainty.modules import PredictionEnsemble
from midnite.uncertainty.modules import PredictiveEntropy
from midnite.uncertainty.modules import VariationRatio

# For consistent tests, seed has to be fixed
torch.manual_seed(123456)


def test_dropout_layer():
    """Prediction dropout layer test

    Checks if the layer does dropout in prediction mode.

    """
    # Set dropout layer to evaluation mode
    dropout_layer = PredictionDropout()
    dropout_layer.eval()

    # Check that dropout still is applied
    input_ = torch.ones((1, 10))
    output = dropout_layer(input_).squeeze(dim=0).numpy()

    assert_that(output).contains(0.0)
    assert_that(output).contains(2.0)
    assert_that(output).contains_only(2.0, 0.0)


def test_mean_ensemble_layer(mocker):
    """Mean ensemble layer test

    Checks if the layer correctly calculates an ensemble mean.

    """
    inner = PredictionDropout()
    mocker.spy(inner, "forward")
    ensemble = MeanEnsemble(inner, 30)
    ensemble.eval()

    input_ = torch.ones((1, 10))
    output = ensemble(input_)

    assert torch.allclose(input_, output, atol=0.34)
    assert not torch.allclose(input_, output)
    assert_that(inner.forward.call_count).is_equal_to(30)


def test_ensemble_layer(mocker):
    """Ensemble layer test

    Checks if the ensemble layer does the sampling correctly.

    """
    inner = PredictionDropout()
    mocker.spy(inner, "forward")
    ensemble = PredictionEnsemble(inner, 25)
    ensemble.eval()

    input_ = torch.ones((1, 10))
    output = ensemble(input_)

    assert_that(output.size()).is_equal_to((1, 10, 25))
    assert torch.all((output == 0.0) + (output == 2.0))
    assert_that(inner.forward.call_count).is_equal_to(25)


def test_ensembles_autograd():
    """Test if autograd is disabled in mean ensemble"""
    inner = PredictionDropout()
    ensemble = PredictionEnsemble(inner)
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_true()

    ensemble.eval()
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_false()

    ensemble = MeanEnsemble(inner)
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_true()

    ensemble.eval()
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_false()


def test_pred_entropy(mocker):
    """Predictive entropy layer test"""
    # Patch out functional
    mocker.patch("midnite.uncertainty.functional.predictive_entropy")

    input_ = torch.ones((1, 20, 2))
    PredictiveEntropy()(input_)

    functional.predictive_entropy.assert_called_once_with(input_)


def test_mutual_information(mocker):
    """Mutual information layer test."""
    mocker.patch("midnite.uncertainty.functional.mutual_information")

    input_ = torch.ones(1, 20, 2)
    MutualInformation()(input_)

    functional.mutual_information.assert_called_once_with(input_)


def test_variation_ratio(mocker):
    """Variation ratio layer test."""
    mocker.patch("midnite.uncertainty.functional.variation_ratio")

    input_ = torch.ones(1, 20, 2)
    VariationRatio()(input_)

    functional.variation_ratio.assert_called_once_with(input_)


def test_prediction_and_uncertainties(mocker):
    """Combined output layer test."""
    mocker.patch("midnite.uncertainty.functional.predictive_entropy")
    mocker.patch("midnite.uncertainty.functional.mutual_information")

    input_ = torch.tensor([[1.0, 0.5, 0.3]])
    output, _, _ = PredictionAndUncertainties()(input_)

    assert_array_almost_equal(output, [0.6])

    functional.predictive_entropy.assert_called_once_with(input_)
    functional.mutual_information.assert_called_once_with(input_)


def test_prediction_ensemble_layer_modes():
    """tests the prediction ensemble, if dropout layers are kept in train mode
     while other layers switch to test mode after eval()"""

    model = torch.nn.Sequential(PredictionDropout(), torch.nn.Conv2d(1, 20, 5))

    ensemble = PredictionEnsemble(model)
    assert_that(ensemble.training).is_true()
    assert_that(ensemble.inner[0].training).is_true()
    assert_that(ensemble.inner[1].training).is_true()
    ensemble.eval()
    assert_that(ensemble.training).is_false()
    assert_that(ensemble.inner[0].training).is_true()
    assert_that(ensemble.inner[1].training).is_false()


def test_mean_ensemble_layer_modes():
    """tests the mean ensemble, if dropout layers are kept in train mode
     while other layers switch to test mode after eval()"""

    model = torch.nn.Sequential(PredictionDropout(), torch.nn.Conv2d(1, 20, 5))

    mean_ensemble = MeanEnsemble(model)
    assert_that(mean_ensemble.training).is_true()
    assert_that(mean_ensemble.inner[0].training).is_true()
    assert_that(mean_ensemble.inner[1].training).is_true()
    mean_ensemble.eval()
    assert_that(mean_ensemble.training).is_false()
    assert_that(mean_ensemble.inner[0].training).is_true()
    assert_that(mean_ensemble.inner[1].training).is_false()
