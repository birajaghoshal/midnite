#!/usr/bin/env python
# coding: utf-8

# # midnite - Visualization Tutorial 1 - Part 1
# 
# in this notebook you will learn the intuition behind the features of the interpretability framework and how to us them.
# 
# ## Gradient Class Activation Mapping (Grad-CAM) with ResNet
# 
# Demonstration of visualizing the class activation mapping for an image classification example with ResNet.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../src')

import data_utils
from data_utils import DataConfig
from plot_utils import *
from PIL import Image

from midnite import get_device
from midnite.visualization.base import *
from midnite.common import Flatten
import torch
from torch import Tensor
from torch.nn import Softmax
from torch.nn.modules import Sequential
from torch.nn.functional import interpolate

from torchvision import models


# ## Step 1: Load pretrained model
# 
# In our example we use a pretrained ResNet for demonstration.

# In[2]:


model = models.resnet50(pretrained=True)

model.eval()
model.to(get_device());


# ## Step 2: Load Image

# In[3]:


input_ = data_utils.get_example_from_path("../data/imagenet_example_283.jpg", DataConfig.ALEX_NET)


# ## Step 3: Select layer of interest
# 
# in ResNet, we have 9 layers, where 8 is the last feature layer and 10 the classification layer
# Input: a list of layers which should be visualized as attribution.
# **Note: choose layers between 1 and 8 for ResNet**

# ### Visualization of layer attributions
# it is possible to visualize attributions of a single layer, or from several layers together. In this example we demonstrate both.

# In[4]:


# example ResNet selection
selected_layer = 8


# ### Gradient-based Class Activation Mapping
# <img src="resources/class-activation-mapping.png">

# ## Step 4: Select layer splits - Class Visualization example
# in this example we want to analyze the top classes of the classification output. 
# This means, we create a top-layer-selector with a Neuronsplit with the class of choice as split element.
# Class 283 is the max classification for the imagenet_example_283. 
# 
# Additionally, it is possible to create a bottom_layer_selector to visualize the class activations in greater detail, e.g. channel-wise or neuron-wise.

# ### Possible splits for the NeuronSelector
# <img src="resources/splits.png">
# source: https://distill.pub/2018/building-blocks/

# In[5]:


top_layer_selector = SplitSelector(NeuronSplit(), [283])
bottom_layer_split = SpatialSplit()


# ## Step 5: Split the model into base_layers and inspection_layers
# splitting the model with classification returns a list of base layers up to the selected single layer and the list of layers (inspection layers) from the selected layer until the last layer of the model, the classification layer. The output of the inspected layers is a classification with dimension (1, 1000)

# In[6]:


base_layers = list(model.children())[:selected_layer]

inspection_layers = (
    list(model.children())[selected_layer:-1]
    + [Flatten(), list(model.children())[-1]]
)


# ## Step 6: Generate Saliency Map
# 
# **Return values for each split:** 
# 
# * NeuronSplit() returns saliency with full dimensions of the selected layer - 3 dimensions (c, h, w)
# * ChannelSplit() returns a saliency value for each channel, hence returns 1 dimension (c)
# * SpatialSplit() returns a saliency value for each spatial, hence returns 2 dimensions (h, w)
# 
# ### Example 1: Compute saliency map with bottom-layer spatial split

# In[7]:


def upsample(img, target):
    return interpolate(
        img.unsqueeze(dim=0).unsqueeze(dim=0), 
        size=target.size()[2:], 
        mode='bilinear',
        align_corners=True
    ).squeeze(0).squeeze(0)


# In[8]:


heatmap = GradAM(
    Sequential(*inspection_layers), 
    top_layer_selector, 
    Sequential(*base_layers), 
    bottom_layer_split
).visualize(input_)

# upsample saliency to the pixel dimensions of the image
heatmap = upsample(heatmap, input_)

# plot saliencies with the input image
show_heatmap(heatmap, 1.5, input_)


# ## Backpropagation

# In[9]:


backprop = torch.nn.functional.relu(Backpropagation(
    Sequential(*base_layers+inspection_layers),
    top_layer_selector,
    bottom_layer_split
).visualize(input_))

show_heatmap(backprop, 1.5)


# # Guided Backpropagation

# In[10]:


backprop = GuidedBackpropagation(
    base_layers+inspection_layers,
    SplitSelector(NeuronSplit(), [283]),
    bottom_layer_split
).visualize(input_)

# plot saliencies with the input image
show_heatmap(backprop, 1.5)


# ### Example 2: Compute saliency map with bottom layer channel split.

# In[11]:


top_layer_selector = SplitSelector(NeuronSplit(), [283])
bottom_layer_split = ChannelSplit()


# In[12]:


heatmap = GradAM(
    Sequential(*inspection_layers), 
    top_layer_selector, 
    Sequential(*base_layers), 
    bottom_layer_split
).visualize(input_)

top_values, top_channels = torch.topk(heatmap, 5)

for (value, idx) in zip(top_values, top_channels):
    print("Channel number: {} with channel value: {}".format(idx, value))


# ### Example 3: Compute saliency map with neuron split
# 

# In[13]:


bottom_layer_split = NeuronSplit()


# In[14]:


heatmap = GradAM(
    Sequential(*inspection_layers), 
    top_layer_selector, 
    Sequential(*base_layers), 
    bottom_layer_split
).visualize(input_)

print(f"Neuron activations for channel 39 of layer {selected_layer}:")

# visualize the neurons of channel 39
heatmap = upsample(heatmap[39], input_)

# plot saliencies neuron activations
show_heatmap(heatmap, 1.5, input_)


# In[ ]:




