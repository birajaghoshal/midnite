#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../src')

import torch
from torch.nn import Sequential
from torch.nn import Softmax
import matplotlib.pyplot as plt
from pytorch_fcn.fcn32s import FCN32s
import data_utils
from data_utils import DataConfig

from interpretability_framework import modules


# # Interpretability-framework - Uncertainty Tutorial 2
# 
# ### Step 1: Load pre-trained FCN32 model

# In[2]:


fully_conf_net = FCN32s()
fully_conf_net.load_state_dict(torch.load(FCN32s.download()))


# ### Step 2: Build and prepare MC ensemble

# In[3]:


ensemble = Sequential(
    modules.PredictionEnsemble(fully_conf_net),
    Softmax(dim=1),
    modules.PredictionAndUncertainties()
)

ensemble.eval();


# ### Step 3: Load example image

# In[4]:


img = data_utils.get_example_from_path("../data/fcn_example.jpg", DataConfig.FCN32)

norm_img = torch.sub(img, img.min())
norm_img = torch.div(norm_img, norm_img.max())

plt.imshow(norm_img.squeeze(dim=0).permute(1, 2, 0))
plt.show()


# ### Step 4: Calculate Uncertainty

# In[5]:


pred, pred_entropy, mutual_info = ensemble(img)
print("Prediction:")
plt.imshow(pred.argmax(dim=1).squeeze(dim=0).detach().numpy())
plt.show()

print("Predictive entropy (total uncertainty):")
plt.imshow(pred_entropy.sum(dim=1).squeeze(dim=0).detach().numpy())
plt.show()

print("Mutual information (model uncertainty):")
plt.imshow(mutual_info.sum(dim=1).squeeze(dim=0).detach().numpy())
plt.show()

