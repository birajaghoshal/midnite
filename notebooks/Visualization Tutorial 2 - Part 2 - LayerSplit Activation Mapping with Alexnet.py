#!/usr/bin/env python
# coding: utf-8

# # vinsight - Visualization Tutorial 1 - Part 2
# 
# in this notebook you will learn the intuition behind the features of the interpretability framework and how to us them.
# 
# ## LayerSplit Activation Mapping with AlexNet
# 
# Demonstration of visualizing layer attribution, the activation mapping  between two specified layers and their splits. We umple with ResNet.
# 
# Additionally, you can test GradCAM AlexNet (see model_utils).

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../src')

import data_utils
from data_utils import DataConfig
from model_utils import ModelConfig
from model_utils import split_model
from PIL import Image
import plot_utils
from vinsight.visualization import SaliencyMap
from vinsight.visualization import SpatialSplit
from vinsight.visualization import ChannelSplit
from vinsight.visualization import NeuronSplit
from vinsight.visualization import NeuronSelector

import torch
from torch import Tensor
from torch.nn import Softmax
from torch.nn.functional import interpolate

from torchvision import models


# ## Step 1: Load pretrained model
# 
# In our example we use a pretrained ResNet for demonstration.

# In[36]:


device = torch.device("cuda" if torch.tensor([]).is_cuda else "cpu")

model = models.alexnet(pretrained=True)

model.eval()
model.to(device);


# ## Step 2: Load Image

# In[3]:


img_path = "../data/imagenet_example_283.jpg"
#img_path = "../data/ood_example.jpg"

# load image as torch tensor for computation
input_ = data_utils.get_example_from_path(img_path, DataConfig.ALEX_NET)

# load image as PIL Image for plotting and resizing
img = Image.open(img_path)
H, W = img.size


# ## Step 3: Select layer of interest
# 
# in AlexNet there are 3 main layers
# 
# ### Single layer visualization
# for example: layer number 10 as single layer.

# In[9]:


selected_layer = 10


# ## Step 3: Select classes of interest

# In[7]:


output_scores, output_classes = torch.topk(Softmax(dim=1)(model(input_).to(device)), 2)


# GradCam of layer 10:

# In[39]:


# for every selected class, generate GradCAM
for i, (score, c) in enumerate(zip(output_scores[0], output_classes[0])):
    
    bottom_layers, top_layers = split_model(model, ModelConfig.ALEX_NET, selected_layer)

    # compute saliency map
    saliency = SaliencyMap(top_layers, bottom_layers, do_split=False, neuron_selector=None).visualize(c, input_)
        
    # upsample saliency to the pixel dimensions of the image
    sal_map = interpolate(saliency.unsqueeze(dim=0).unsqueeze(dim=0), size=(H, W), mode='bilinear', align_corners=True)
    
    plot_utils.plot_saliency(sal_map, img, c, score, selected_layer)


# ## Step 4: Select a layer split
# if you want to do layer split, 
# * set do_split on True, 
# * select an element of choice
# * select a layer split with a neuron selector
# 
# ### Option 1: Channel Split
# inspecting channel 100

# In[33]:


do_split = True
selected_channel = [100]
neuron_selector = NeuronSelector(ChannelSplit(), selected_channel)

# for every selected class, generate GradCAM
for i, (score, c) in enumerate(zip(output_scores[0], output_classes[0])):
    
    bottom_layers, top_layers = split_model(model, ModelConfig.ALEX_NET, selected_layer)
        
    # compute saliency map
    saliency = SaliencyMap(top_layers, bottom_layers, do_split, neuron_selector).visualize(c, input_)
        
    # upsample saliency to the pixel dimensions of the image
    sal_map = interpolate(saliency.unsqueeze(dim=0).unsqueeze(dim=0), size=(H, W), mode='bilinear', align_corners=True)
    
    plot_utils.plot_saliency(sal_map, img, c, score, selected_layer)


# ### Option 2: Spatial Split
# inspecting spatial 3, 3

# In[25]:


do_split = True
selected_spatial = [3, 3]
neuron_selector = NeuronSelector(SpatialSplit(), selected_spatial)


# In[26]:


# for every selected class, generate GradCAM
for i, (score, c) in enumerate(zip(output_scores[0], output_classes[0])):
    
    bottom_layers, top_layers = split_model(model, ModelConfig.ALEX_NET, selected_layer)
    
    # compute saliency map
    saliency = SaliencyMap(top_layers, bottom_layers, do_split, neuron_selector).visualize(c, input_)
        
    # upsample saliency to the pixel dimensions of the image
    sal_map = interpolate(saliency.unsqueeze(dim=0).unsqueeze(dim=0), size=(H, W), mode='bilinear', align_corners=True)
    
    plot_utils.plot_saliency(sal_map, img, c, score, selected_layer)
    


# In[34]:


selected_spatial = [5, 5]
neuron_selector = NeuronSelector(SpatialSplit(), selected_spatial)


# In[35]:


# for every selected class, generate GradCAM
for i, (score, c) in enumerate(zip(output_scores[0], output_classes[0])):
    
    bottom_layers, top_layers = split_model(model, ModelConfig.ALEX_NET, selected_layer)
    
    # compute saliency map
    saliency = SaliencyMap(top_layers, bottom_layers, do_split, neuron_selector).visualize(c, input_)
        
    # upsample saliency to the pixel dimensions of the image
    sal_map = interpolate(saliency.unsqueeze(dim=0).unsqueeze(dim=0), size=(H, W), mode='bilinear', align_corners=True)
    
    plot_utils.plot_saliency(sal_map, img, c, score, selected_layer)

