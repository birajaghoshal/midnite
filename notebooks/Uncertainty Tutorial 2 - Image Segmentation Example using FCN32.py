#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from vinsight.uncertainty import modules


# # Interpretability-framework - Uncertainty Tutorial 2
# in this notebook you will learn how to use the interpretability framework in an image segmentation example and how to interpret the results.
# 
# ## Segmentation example with FCN32s
# Demonstration of uncertainty measurement and visualization in an segmentation example.
# 
# 
# ### Step 1: Load pre-trained FCN32 model
# **Note: Monte Carlo Dropout ensembles only achieve proper results when used on a net which was trained with dropout. So check if the model you would like to use has dropout layers active in training.**

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


# ## Interpretation of results 
# In contrast to the image classification example, we calculate predictions and uncertainties for each pixel of the image. 
# Hence, we can visualize the pixel uncertainties as an image and add a color scheme to make interpretations easier (red = high uncertainty, blue = low uncertainty).
# 
# What can we see in these uncertainty images?
# 
# * **Total predictive entropy:** the total predictive uncertainty of each pixel, we can detect which pixels have high uncertainty and therefore, are predicted with low confidence.
# * **Mutual Information:** the model uncertainty of each pixel, we can detect which pixels are predicted with low confidence due to the model.
# * **Data Uncertainty:** we can detect which pixels are predicted with low confidence due to the quality of the data (image).
