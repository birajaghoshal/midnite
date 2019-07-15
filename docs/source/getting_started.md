# Getting started
Read here on how to use midnite.
If you prefer interaction with the code instead of explanations, check out the [notebooks](notebooks/midnite.md)!

## Installation
Install the package `midnite` from [pypi](https://pypi.org/) like this:

```bash
pip install midnite==0.1.0
```

for more detailed installation instructions, view the project's [README](https://gitlab.com/luminovo/public/midnite/blob/master/README.md).

## Device context
When using midnite, it's often helpful to run torch on a specific device.
Since it's a framework and no application, we can't just start it with a flag where to run.

Instead, there is a context for the computation device:
```python
import midnite

with midnite.device('cuda:0'):
    # do some cool midnite stuff...
```

If you want to know in which context you're currently in, that's also simple:

```python
import midnite

torch_device = midnite.get_device()

# do stuff with device
```

## Midnite core
In its core, `midnite` consists uncertainty and visualization methods. 
Read on how to use them:
```eval_rst
.. toctree::
   :maxdepth: 2
   
   uncertainty
   visualization
```

## Common layers
Some common layers are unfortunately not provided by torch, but you can find them here.
 - `midnite.common.Flatten`: Flattens a convolutional layer into a FCN input.
   If you're wondering why this isn't in torch, see [this issue](https://github.com/pytorch/pytorch/issues/2118).