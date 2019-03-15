#!/usr/bin/env python
# coding: utf-8

# # Interpretability-framework - Uncertainty Tutorial 1
# 
# in this notebook you will learn the intuition behind the features of the interpretability framework and how to us them.

# ## Classification Example with AlexNet
# 
# Demonstration of an uncertainty measurement of an image classification example.

# In[1]:


import torchvision.models as models
from torch.nn import Dropout
from torch.nn import Softmax
from torch.nn import Sequential

import data_utils
from data_utils import DataConfig
from interpretability_framework import modules


# ### Step 1:
# 
# Load the model, which should be evaluated.
# 
# **Note: Monte Carlo Dropout ensembles only achieve proper results when used on a net which was trained with dropout. So check if the model you would like to use has dropout layers active in training.**
# 
# In our example we use a pretrained AlexNet for demonstration.

# In[2]:


alexnet = models.alexnet(pretrained=True)
alexnet.classifier.add_module("softmax", Softmax(dim=1))


# ### Step 2:
# 
# Wrap your net with an PredictionEnsemble layer. This layer collects an ensemble of predictions with Monte Carlo dropout. This ensemble will be used for measuring uncertainties. 

# In[3]:


ensemble = Sequential(
        modules.PredictionEnsemble(inner=alexnet), modules.ConfidenceMeanPrediction()
    )


# ### Step 3:
# 
# Prepare the ensemble layer by entering evaluation mode, to specify, that we are in test mode.

# In[ ]:


ensemble.eval()


# ### Step 4: 
# 
# You can evaluate the predictive uncertainties by loading an image from a path or a batch of a dataset.
# data.utils transforms the images, as AlexNet needs examples with dimensions ( 1 x 3 x 227 x 227 ).
# 
# 
# **Option 1: use batch data** 
# 
# We can retrieve a batch of data from a dataset by using the pytorch dataloader implemented in data.utils.py.
# Let's specify a batch size of 5 examples.

# In[ ]:


# download dataset first!
# batch = load_imagenet_dataset("../data/imagenet")


# **Option 2: use image from path**
# 
# Load an image example, for which the uncertainties should be calculated.
# 
# *In order to interpret the results, we want to compare the uncertainties of an in-distribution and an out-of-distribution example.* 
# As AlexNet was trained with ImageNet data, we pick an ImageNet example as an in-distribution target, for which predictions should be very confident. An out-of-distribution target is an image, which depicts something, which is not a class label of ImageNet. We use Cholitas for this example.

# In[ ]:


id_example = data_utils.get_example_from_path("../data/imagenet_example_283.jpg", DataConfig.ALEX_NET)
ood_example = data_utils.get_example_from_path("../data/ood_example.jpg", DataConfig.ALEX_NET)


# **Option 3: generate a random example**
# 
# Retrieve a random example normalized with the distribution mean and standard deviation, which can also be seen as an out-of-distribution sample.
# 

# In[ ]:


random_example = data_utils.get_random_example(DataConfig.ALEX_NET)


# ### Step 6: 
# Calculate uncertainties.

# In[ ]:


pred, pred_entropy, mutual_info = ensemble(id_example)

print("Uncertainties of in-distribution example")
print(f"    mean prediction: {pred.argmax()}, class probability: {pred.max()}")
print(f"    total predictive entropy: {pred_entropy.sum()}")
print(f"    total mutual information: {mutual_info.sum()}")


# In[ ]:


pred, pred_entropy, mutual_info = ensemble(ood_example)

print("Uncertainties of out-of-distribution example")
print(f"    mean prediction: {pred.argmax()}, class probability: {pred.max()}")
print(f"    total predictive entropy: {pred_entropy.sum()}")
print(f"    total mutual information: {mutual_info.sum()}")


# In[ ]:


pred, pred_entropy, mutual_info = ensemble(random_example)

print("Uncertainties of the random example")
print(f"    mean prediction: {pred.argmax()}, class probability: {pred.max()}")
print(f"    total predictive entropy: {pred_entropy.sum()}")
print(f"    total mutual information: {mutual_info.sum()}")


# ## Interpretation of results
# 
# We should see that in-distribution examples have a class probability close to 1 and a low uncertainties. Which means, we are predicting with high confidence.
# 
# In the OOD and random sample we see that the prediction can be different everytime the ensemble is executed. This is due to the stochasticity of the MC dropout. 
# 
# Still, we can observe that the uncertainty is constantly very high, so we predict with low confidence. Whereas, the total predictive entropy (total uncertainty) is high for both, OOD and random example, we see that the mutual information is significantly different. High mutual information idicates that the total uncertainty is mainly due to model uncertainty. Whereas when the mutual information is low, it is an indicator that the total uncertainty is due to data uncertainty.
# 
# 
