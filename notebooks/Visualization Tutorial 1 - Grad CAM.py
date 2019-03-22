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

# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../src')

import data_utils
from data_utils import DataConfig
from PIL import Image
import plot_utils
from vinsight.visualization import SaliencyMap
from vinsight.visualization import Flatten
from vinsight.visualization import SpatialSplit

import torch
from torch import Tensor
from torch.nn import Softmax

from torchvision import models


# In[16]:


model = models.resnet50(pretrained=True)

model.eval()


# ## Load Image

# In[17]:


img_path = "../data/imagenet_example_283.jpg"
input_ = data_utils.get_example_from_path(img_path, DataConfig.ALEX_NET)


# 1) dimensionality split - select layersplit
# 
# 2) compute a forward pass to retrieve features (input)
# 
# 3) compute forward pass through layer of choice to retrieve output
# 

# ## select layer of interest
# 
# in ResNet, we have 9 layers, where 8 is the last feature layer and 10 the classification layer
# 
# **Note: please choose a layer between 0 and 8** #TODO method: check if choice is valid

# In[18]:


selected_layer = 7


# ## select class of interest
# in this example we use the topk classes and compute their saliency maps

# In[22]:


output_scores, output_classes = torch.topk(Softmax(dim=1)(model(input_)), 2)
img = Image.open(img_path)

bottom_layers = list(model.children())[:selected_layer]
top_layers = (list(model.children())[selected_layer:-1] + [Flatten()] + list(model.children())[-1:])
bottom_layer_split = SpatialSplit()

for i, (score, c) in enumerate(zip(output_scores[0], output_classes[0])):
    saliency_map = SaliencyMap(top_layers, bottom_layers, bottom_split).visualize(c, input_)
    plot_utils.plot_saliency(saliency_map, img, c, score, selected_layer)


# In[ ]:




