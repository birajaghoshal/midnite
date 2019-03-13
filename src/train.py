"""Main script. Runs some examples on how to use the framework."""
import torch
import torchvision.models as models
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import Softmax

import data_utils
from interpretability_framework import modules


def main():
    """Example on how to use uncertainty measures for existing model."""
    # Activate cuda if available
    if torch.cuda.is_available and torch.cuda.device_count() > 0:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Get pre-trained model from torchvision
    alexnet = models.alexnet(pretrained=True)
    alexnet.classifier.add_module("softmax", Softmax(dim=1))

    # MC Ensemble that calculates the mean, predictive entropy, mutual information,
    # and variation ratio.
    ensemble = Sequential(
        modules.PredictionEnsemble(alexnet, sample_size=50),
        modules.ConfidenceMeanPrediction(),
    )

    # Prepare ensemble
    ensemble.eval()

    # Re-activate dropout layers
    for layer in list(alexnet.modules()):
        if isinstance(layer, Dropout):
            layer.train()

    # Get input images from dataloader, torch tensor of dim: (1 x 3 x 227 x 227)
    for img_name, img in [
        (
            "Imagenet example",
            data_utils.get_example_from_path("../data/imagenet_example.jpg"),
        ),
        (
            "Other dataset example",
            data_utils.get_example_from_path("../data/ood_example.jpg"),
        ),
        ("Random pixels", data_utils.get_random_example()),
    ]:
        # Do prediction
        pred, pred_entropy, mutual_info, var_ratio = ensemble(img)
        print(f"{img_name}:")
        print(f"    mean prediction: {pred.argmax()}, class probability: {pred.max()}")
        print(f"    total predictive entropy (total uncertainty): {pred_entropy.sum()}")
        print(f"    total mutual information (model uncertainty): {mutual_info.sum()}")
        print(f"    variational ratio (uncertainty percentage): {var_ratio.item()}")


if __name__ == "__main__":
    main()
