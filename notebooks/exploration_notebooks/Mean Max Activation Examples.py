#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../../src')

from torchvision import models
from matplotlib import pyplot as plt
import numpy as np
import torch

from vinsight.visualization import PixelActivationOpt, NeuronSelector, ChannelSplit, SpatialSplit, NeuronSplit


# In[3]:


alexnet = models.alexnet(pretrained=True)


# # Channel-wise

# In[6]:


res = PixelActivationOpt(alexnet.features[:9], NeuronSelector(ChannelSplit(), [1])).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()


# In[36]:


res = PixelActivationOpt(alexnet.features[:11], NeuronSelector(ChannelSplit(), [0])).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()


# In[90]:


res = PixelActivationOpt(alexnet.features[:13], NeuronSelector(ChannelSplit(), [0])).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()


# # Spatially

# In[45]:


res = PixelActivationOpt(alexnet.features[:11], NeuronSelector(SpatialSplit(), [3, 3]),size=100,steps=100).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()


# # Individual Neurons

# In[29]:


res = PixelActivationOpt(alexnet.features[:11], NeuronSelector(NeuronSplit(), [0, 0, 0]),size=100,steps=100).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()


# In[70]:


res = PixelActivationOpt(alexnet.features[:4], NeuronSelector(NeuronSplit(), [1, 0, 0]),size=100,steps=10,lr=1).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()


# In[5]:


res = PixelActivationOpt(alexnet.features[:6], NeuronSelector(NeuronSplit(), [5, 0, 0]),size=100,steps=30).visualize().numpy()
res = res - res.min()
res = res / res.max()
plt.imshow(res)
plt.show()

