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


def test_mean_ensemble_layer():
    """Ensemble layer test

    Tests if the mean ensemble layer correctly calculates an ensemble mean.

    """
    inner = modules.PredDropout()
    ensemble = modules.MeanEnsemble(inner, 20)
    ensemble.eval()

    input = torch.ones(10)
    output = ensemble.forward(input)

    assert torch.allclose(input, output, atol=0.3)
    assert not torch.allclose(input, output)


def test_ensemble_layer():
    inner = modules.PredDropout()
    ensemble = modules.PredictionEnsemble(inner, 20)
    ensemble.eval()

    input = torch.ones(10)
    output = ensemble.forward(input)

    assert list(output.size()) == [10, 20]
    assert torch.all((output == 0.0) + (output == 2.0))

def test_pred_entropy():
    inner = modules.PredDropout()
    ensemble = modules.MeanEnsemble(inner, 20)
    ensemble.eval()

    input = torch.ones(10)
    pred = ensemble.forward(input)

    entropy = modules.PredictiveEntropy()
    pred_entropy = entropy.forward(pred)

    assert list(pred_entropy.size()) == [10]
    assert float(pred_entropy.max()) <= 1.0 and float(pred_entropy.min()) >= -0.5

