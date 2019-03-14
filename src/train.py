"""Main script. Runs some examples on how to use the framework."""
import torch
from torch.nn import Sequential
from torch.nn import Softmax

from interpretability_framework import modules
from pytorch_fcn.fcn32s import FCN32s


def uncertainty_segmentation():
    # Load pre-trained model
    fully_conf_net = FCN32s()
    fully_conf_net.load_state_dict(torch.load(FCN32s.download()))

    ensemble = Sequential(
        modules.PredictionEnsemble(fully_conf_net, 20),
        modules.PredictiveEntropy(),
        Softmax(dim=1),
    )

    ensemble.eval()


def main():
    uncertainty_segmentation()


if __name__ == "__main__":
    main()
