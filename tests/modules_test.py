import torch

from interpretability_framework import functional as F
from interpretability_framework import modules

# For consistent tests, seed has to be fixed
torch.manual_seed(123456)


def test_dropout_layer():
    """Prediction dropout layer test

    Checks if the layer does dropout in prediction mode.

    """
    # Set dropout layer to evaluation mode
    dropout_layer = modules.PredDropout(p=0.5)
    dropout_layer.eval()

    # Check that dropout still is applied
    input_ = torch.ones(10)
    output = dropout_layer(input_)

    assert torch.all((output == 2.0) + (output == 0.0))
    assert not torch.all(output == 0.0)
    assert not torch.all(output == 2.0)


def test_mean_ensemble_layer():
    """Mean ensemble layer test

    Checks if the layer correctly calculates an ensemble mean.

    """
    inner = modules.PredDropout()
    ensemble = modules.MeanEnsemble(inner, 20)
    ensemble.eval()

    input_ = torch.ones(10)
    output = ensemble(input_)

    assert torch.allclose(input_, output, atol=0.3)
    assert not torch.allclose(input_, output)


def test_ensemble_layer():
    """Ensemble layer test

    Tests if the ensemble layer does the sampling correctly.

    """
    inner = modules.PredDropout()
    ensemble = modules.PredictionEnsemble(inner, 20)
    ensemble.eval()

    input_ = torch.ones(10)
    output = ensemble(input_)

    assert list(output.size()) == [20, 10]
    assert torch.all((output == 0.0) + (output == 2.0))


def test_pred_entropy(mocker):
    """Predictive entropy layer test

    """
    # Patch out functional
    mocker.patch("interpretability_framework.functional.predictive_entropy")

    input_ = torch.ones((20, 2))
    modules.PredictiveEntropy()(input_)

    F.predictive_entropy.assert_called_once_with(input_)


def test_mutual_information(mocker):
    """Mutual information layer test.

    """
    mocker.patch("interpretability_framework.functional.mutual_information")

    input_ = torch.ones(20, 2)
    modules.MutualInformation()(input_)

    F.mutual_information.assert_called_once_with(input_)


def test_variation_ratio(mocker):
    """Variation ratio layer test.

    """
    mocker.patch("interpretability_framework.functional.variation_ratio")

    input_ = torch.ones(20, 2)
    modules.VariationRatio()(input_)

    F.variation_ratio.assert_called_once_with(input_)
