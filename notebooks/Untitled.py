#!/usr/bin/env python
# coding: utf-8

# # vinsight - Visualization Tutorial 1
# 
# in this notebook you will learn the intuition behind the features of the interpretability framework and how to us them.
# 
# ## Gradient Class Activation Mapping (Grad-CAM) with ResNet
# 
# Demonstration of visualizing the class activation mapping for an image classification example.
# 
# ### Step 1: Load pretrained model
# 
# In our example we use a pretrained ResNet for demonstration.
# 

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../src')

import data_utils
from data_utils import DataConfig
from vinsight.visualization import building_blocks
import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Softmax

from torchvision import models
from matplotlib import pyplot as plt


# In[10]:


model = models.resnet50(pretrained=True)

# split model for activation mapping
features_fn = Sequential(*list(model.children())[:-2])
classifier_fn = Sequential(*(list(model.children())[-2:-1] + [building_blocks.Flatten()] + list(model.children())[-1:]))

model.eval()


# Load Image

# In[12]:


img_path = "../data/imagenet_example_283.jpg"
input_ = data_utils.get_example_from_path(img_path, DataConfig.ALEX_NET)


# 1) dimensionality split and select neuron
# 
# 2) compute a forward pass to retrieve features (input)
# 
# 3) compute forward pass through layer of choice to retrieve output
# 

# In[ ]:




