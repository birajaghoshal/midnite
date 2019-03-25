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

from vinsight.visualization import *

if torch.cuda.is_available and torch.cuda.device_count() > 0:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


# In[2]:


alexnet = models.alexnet(pretrained=True)


# # Channel-wise
# ## No regularization

# In[3]:


res = PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[4]:


res = PixelActivationOpt(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    iter_transform = BiliteralTransform()
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[4]:


res = PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    iter_transform = RandomTrans()
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[5]:


res = PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    iter_transform = RandomTrans() + BiliteralTrans() + ResizeTrans(),
    init_size = 50
).visualize().numpy()
plt.imshow(res)
plt.show()


# ## Blur Regularization

# In[42]:


res = PixelActivationOpt(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    iter_transform = BlurTransformStep(),
).visualize().numpy()
plt.imshow(res)
plt.show()


# ## Resizing regularization

# In[43]:


res = PixelActivationOpt(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    iter_transform = BlurResizeStep(),
    init_size=50
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[37]:


res = PixelActivation(
    alexnet.features[:9],
    NeuronSelector(ChannelSplit(), [1]),
    reg=TVReg(),
    transform=BlurTrans()
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[38]:


res = PixelActivation(
    alexnet.features[:11],
    NeuronSelector(ChannelSplit(), [0]),
    reg=TVReg(coefficient=0.9)
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[ ]:





# In[47]:


res = PixelActivationOpt(
    alexnet.features[:11],
    NeuronSelector(ChannelSplit(), [0]),
    iter_transform = BlurTransformStep(),
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[48]:


res = PixelActivationOpt(
    alexnet.features[:13],
    NeuronSelector(ChannelSplit(), [0]),
    iter_transform = BlurTransformStep(),
).visualize().numpy()
plt.imshow(res)
plt.show()


# # Spatially

# In[49]:


res = PixelActivationOpt(
    alexnet.features[:11],
    NeuronSelector(SpatialSplit(), [3, 3]),
    iter_transform = BlurTransformStep(),
    init_size=100,
    steps_per_iter=100
).visualize().numpy()

plt.imshow(res)
plt.show()


# # Individual Neurons

# In[50]:


res = PixelActivationOpt(
    alexnet.features[:11],
    NeuronSelector(NeuronSplit(), [0, 0, 0]),
    init_size=100,
    steps_per_iter=100,
    iter_transform = BlurTransformStep(),
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[51]:


res = PixelActivationOpt(
    alexnet.features[:4],
    NeuronSelector(NeuronSplit(), [1, 0, 0]),
    iter_transform = BlurTransformStep(),
    init_size=100,
    steps_per_iter=10,
    lr=1
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[52]:


res = PixelActivationOpt(
    alexnet.features[:6],
    NeuronSelector(NeuronSplit(), [5, 0, 0]),
    iter_transform = BlurTransformStep(),
    init_size=100,
    steps_per_iter=30
).visualize().numpy()
plt.imshow(res)
plt.show()


# # Class visualization

# In[ ]:


res = PixelActivationOpt(
    [alexnet],
    NeuronSelector(NeuronSplit(), [783]),
    iter_transform = BlurResizeStep(),
    init_size=50,
    weight_decay=1e-6,
    num_iters=30,
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[12]:


res = PixelActivationOpt(
    [alexnet],
    NeuronSelector(NeuronSplit(), [783]),
    iter_transform = BiliteralTransform() + ResizeTransform(),
    num_iters=14,
    init_size=50,
    steps_per_iter=100,
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[11]:


res_ = res - res.min()
res_ = res_ / res_.max()
plt.imshow(res_)
plt.show()


# In[43]:


res = PixelActivation(
    [alexnet],
    NeuronSelector(NeuronSplit(), [783]),
    transform = RandomTrans() + ResizeTrans() + BilateralTrans(),
    reg=TVReg(coefficient=0.9),
    iter_n=12,
    opt_n=50,
    init_size=50
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[46]:


res = PixelActivation(
    [alexnet],
    NeuronSelector(NeuronSplit(), [783]),
    transform = RandomTrans() + BlurTrans(),
    reg=TVReg(coefficient=0.1),
    iter_n=30
).visualize().numpy()
plt.imshow(res)
plt.show()


# In[ ]:




