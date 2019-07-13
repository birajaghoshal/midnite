# Uncertainty
To assess the uncertainty of our model and our predictions, we add _stochasticity_ to the model and evaluate it for a probability distribution.
In particular, for non-bayesian CNNs, we approximate the predictive distribution by sampling over multiple predictions while performing dropout at test time (Monte-Carlo (MC) dropout).

**Note**: This requires that your model is trained with dropout.

To evaluate uncertainty for an existing model, implement the `StochasticModule` interface and create an `Ensemble` for your model.
See the [notebook](notebooks/1_monte_carlo_uncertainty) for a hands-on.

## Implementing `StochasticModule`
The [`StochasticModule`](api/midnite.uncertainty.modules.rst#midnite.uncertainty.modules.StochasticModule) interface allows your network to switch into the 'stochastic evaluation' mode needed for MC dropout.
If your model only uses _Dropout_ layers from torch (`torch.nn.Dropout`, `torch.nn.Dropoud2d`, etc.), you can simply use the [`StochasticDropouts`](api/midnite.uncertainty.modules.rst#midnite.uncertainty.modules.StochasticDropouts) wrapper:
```python
my_stochastic_model = StochasticDropouts(model)
```

Otherwise, you have to implement the interface - here's an example for the torchvision _DenseNet121_:
```python
class StochasticDenseNet121(StochasticModule, DenseNet121):
    def stochastic_eval(self):
        super().stochastic_eval()
        # Layer 4, 6, 8, 10 use dropout when dl.training is set to true
        for i in [4, 6, 8, 10]:
            for dl in self.cnn.features[i]:
                dl.training = True
```

## Create a prediciton ensemble
Now that you have a `StochasticModule`, you can build a MC Dropout ensemble.
It consists of the following parts:
1. [`EnsembleBegin`](api/midnite.uncertainty.modules.rst#midnite.uncertainty.modules.EnsembleBegin): splits the prediction up into multiple (by default: 20) samples
2. [`EnsembleLayer`](api/midnite.uncertainty.modules.rst#midnite.uncertainty.modules.EnsembleLayer): layer that is executed for each sample
3. [Acquisition function](#acquisition-functions): calculates some measure of uncertainty for the sampled predictions, e.g. [`PredictiveEntropy`](api/midnite.uncertainty.modules.rst#midnite.uncertainty.modules.PredictiveEntropy)
 
```python
ensemble = Ensemble(
    EnsembleBegin(),
    PredictiveEntropy(),
    EnsembleLayer(my_stochastic_model),
)

```

Now all that's left is to activate the stochastic eval mode and make some predictions!
```python
ensemble.stochastic_eval()

with midnite.device('cuda:0'):
    uncertainty = ensemble(input_image)
```

## Understanding uncertainty
Now that we can calculate uncertainty, let's have a look at what it actually means:

### Taxonomy
```eval_rst
When talking about uncertainty, we differentiate between *epistemic* and *aleatoric* uncertainty. :cite:`a-uncertainty_gal2016uncertainty`

- *epistemic* uncertainty captures the model uncertainty, i.e. the uncertainty of the learned model parameters
- *aleatoric* uncertainty is the uncertainty that is inherent to the data, e.g., noise or measurement imperfections
```
Also, some literature further distinguishes _distribution uncertainty_, i.e. the mismatch between the distribution in the training and test dataset.

### Acquisition Functions
Those model uncertainties are not independent from each other and not all can be calculated directly.
We implement the following approximations:
 - _predictive entropy_: captures the _total_ uncertainty in the prediction
 - _mutual information_ based entropy: captures model uncertainty (and, depending how you view it, also distribution uncertainty)
 - _variation ratio_: gives point estimation for total uncertainty on a 0-1 scale (simply `1 - max class probability`)


For more information on this topic, see our [references](references.md).
```eval_rst
.. bibliography:: references.bib
   :labelprefix: A
   :keyprefix: a-
   :style: plain
```