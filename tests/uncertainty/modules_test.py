"""Tests for the custom nn modules"""
import torch
from assertpy import assert_that
from numpy.testing import assert_array_almost_equal

from vinsight.uncertainty import functional
from vinsight.uncertainty import modules

# For consistent tests, seed has to be fixed
torch.manual_seed(123456)


def test_dropout_layer():
    """Prediction dropout layer test

    Checks if the layer does dropout in prediction mode.

    """
    # Set dropout layer to evaluation mode
    dropout_layer = modules.PredDropout()
    dropout_layer.eval()

    # Check that dropout still is applied
    input_ = torch.ones((1, 10))
    output = dropout_layer(input_).squeeze(dim=0).numpy()

    assert_that(output).contains(0.0)
    assert_that(output).contains(2.0)
    assert_that(output).contains_only(2.0, 0.0)


def test_mean_ensemble_layer():
    """Mean ensemble layer test

    Checks if the layer correctly calculates an ensemble mean.

    """
    inner = modules.PredDropout()
    ensemble = modules.MeanEnsemble(inner)
    ensemble.eval()

    input_ = torch.ones((1, 10))
    output = ensemble(input_)

    assert torch.allclose(input_, output, atol=0.3)
    assert not torch.allclose(input_, output)


def test_ensemble_layer():
    """Ensemble layer test

    Checks if the ensemble layer does the sampling correctly.

    """
    inner = modules.PredDropout()
    ensemble = modules.PredictionEnsemble(inner)
    ensemble.eval()

    input_ = torch.ones((1, 10))
    output = ensemble(input_)

    assert_that(output.size()).is_equal_to((1, 10, 20))
    assert torch.all((output == 0.0) + (output == 2.0))


def test_ensembles_autograd(mocker):
    """Test if autograd is disabled in mean ensemble"""
    inner = modules.PredDropout()
    ensemble = modules.PredictionEnsemble(inner)
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_true()

    ensemble.eval()
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_false()

    ensemble = modules.MeanEnsemble(inner)
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_true()

    ensemble.eval()
    out = ensemble(torch.zeros(1, requires_grad=True))
    assert_that(out.requires_grad).is_false()


def test_pred_entropy(mocker):
    """Predictive entropy layer test"""
    # Patch out functional
    mocker.patch("vinsight.uncertainty.functional.predictive_entropy")

    input_ = torch.ones((1, 20, 2))
    modules.PredictiveEntropy()(input_)

    functional.predictive_entropy.assert_called_once_with(input_)


def test_mutual_information(mocker):
    """Mutual information layer test."""
    mocker.patch("vinsight.uncertainty.functional.mutual_information")

    input_ = torch.ones(1, 20, 2)
    modules.MutualInformation()(input_)

    functional.mutual_information.assert_called_once_with(input_)


def test_variation_ratio(mocker):
    """Variation ratio layer test."""
    mocker.patch("vinsight.uncertainty.functional.variation_ratio")

    input_ = torch.ones(1, 20, 2)
    modules.VariationRatio()(input_)

    functional.variation_ratio.assert_called_once_with(input_)


def test_prediction_and_uncertainties(mocker):
    """Combined output layer test."""
    mocker.patch("vinsight.uncertainty.functional.predictive_entropy")
    mocker.patch("vinsight.uncertainty.functional.mutual_information")

    input_ = torch.tensor([[1.0, 0.5, 0.3]])
    output, _, _ = modules.PredictionAndUncertainties()(input_)

    assert_array_almost_equal(output, [0.6])

    functional.predictive_entropy.assert_called_once_with(input_)
    functional.mutual_information.assert_called_once_with(input_)
