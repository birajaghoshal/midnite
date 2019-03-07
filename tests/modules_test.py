import torch

from interpretability_framework import modules

# For consistent tests, seed has to be fixed
torch.manual_seed(123456)


def test_dropout_layer():
    """Prediction dropout layer test

    Tests if the layer does dropout in prediction mode.

    """
    # Set dropout layer to evaluation mode
    dropout_layer = modules.PredDropout(p=0.5)
    dropout_layer.eval()

    # Check that dropout still is applied
    input = torch.ones(10)
    output = dropout_layer(input)

    assert not torch.allclose(input, output)
    assert not torch.allclose(input + input, output)


def test_ensemble_layer():
    """Ensemble layer test

    Tests if the mean ensemble layer correctly calculates an ensemble mean.

    """
    inner = modules.PredDropout()
    ensemble = modules.EnsembleMean(inner, 20)
    ensemble.eval()

    input = torch.ones(10)
    output = ensemble.forward(input)

    assert torch.allclose(input, output, atol=0.3)
    assert not torch.allclose(input, output)
