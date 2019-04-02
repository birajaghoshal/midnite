#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../../src')

from torchvision import models
from torch.nn.modules import Softmax
from matplotlib import pyplot as plt
import numpy as np
import torch
from plot_utils import show

import vinsight
from vinsight.visualization import *


# In[2]:


alexnet = models.alexnet(pretrained=True)


# # 1. Channel-wise regularization comparisons
# Here we will look at the effect of different regularizers.
# ## 1.1 No regularization
# Per default, some weight decay is used, which can be seen as l1-reg. Hence we set it to zero in most of the following experiments.

# In[3]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=0
).visualize())


# ## 1.2 Weight decay (l1)
# Less relevant parts of the image are set to zero.

# In[4]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=5e-6
).visualize())


# ## 1.3 Blur Filter
# Performs simple blurring after each step. Has the issue that edges are not preserved.

# In[5]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=0,
    transform = BlurTrans()
).visualize())


# ## 1.4 Bilateral Filter
# Like blur, but preserves edges.

# In[6]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=0,
    transform=BilateralTrans()
).visualize())


# ## 1.5 Random robustness transformations
# Apply random translation, rotation, and scaling after each iteration

# In[7]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=0,
    transform=RandomTrans()
).visualize())


# ## 1.6 Resizing transform
# After each iteration, scale the image up. This has the advantage that low-frequency patterns can be picked up better.

# In[8]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=0,
    transform=ResizeTrans(),
    init_size=50
).visualize())


# ## 1.7 Total Variation Regularizer
# Add total variation to the loss, which punishes difference in adjacent pixels.

# In[9]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    weight_decay=0,
    reg=TVReg(coefficient=0.5)
).visualize())


# # 2. Combining regularizers
#  - weight decay to gray out irrelevant parts
#  - blur (or biliteral filter) for penalizing high-frequency noise
#  - resizing to capture low-frequency patterns
#  - random transformations to get robust image
#  - total variation to get more natural looking image

# In[10]:


show(PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    transform=RandomTrans()+BlurTrans()+ResizeTrans(),
    weight_decay=3e-7,
    init_size=50,
    reg=TVReg()
).visualize())


# # 3. Spatial and Individual Splits
# ## 3.1 Spatial
# Note how some parts of the image do not have any influence on the gradient, since the convolutions have not yet progressed as far.

# In[11]:


show(PixelActivation(
    alexnet.features[:6],
    NeuronSelector(SpatialSplit(), [3, 3]),
    transform=BlurTrans(),
    weight_decay=1e-6,
).visualize())


# # 3.2 Individual Neurons

# In[12]:


show(PixelActivation(
    alexnet.features[:11],
    NeuronSelector(NeuronSplit(), [0, 0, 0]),
    transform=BlurTrans(),
).visualize())


# # 4. Class visualization
# Try to visualize class 783 (Screw)
# 
# TODO experiment (using a GPU) and add result here
