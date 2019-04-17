#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '../src')

import torch
import torchvision.models as models
from torch.nn import Dropout
from torch.nn import Softmax
from torch.nn import Sequential
import matplotlib.pyplot as plt
import data_utils
from data_utils import DataConfig
from plot_utils import show_normalized
from plot_utils import *

from midnite.uncertainty import modules


# In[2]:


alexnet = models.alexnet(pretrained=True)
alexnet.classifier.add_module("softmax", Softmax(dim=1))


# In[3]:


id_example = data_utils.get_example_from_path("../data/imagenet_example_283.jpg", DataConfig.ALEX_NET)
ood_example = data_utils.get_example_from_path("../data/ood_example.jpg", DataConfig.ALEX_NET)

show(id_example)
show(ood_example)


# In[4]:


random_example = data_utils.get_random_example(DataConfig.ALEX_NET)


# In[5]:


from midnite.visualization import graduam
from torch.nn import Sequential
from midnite.common import Flatten
import midnite


def uncert(img):
    with midnite.device("cuda:0"):
        result = graduam(Sequential(*(list(alexnet.features.children()) + [alexnet.avgpool])),
                         Sequential(*([Flatten()] + list(alexnet.classifier.children()))),
                        img)
        show_heatmap(result, 1.5)
        show_heatmap(result, 1.5, img)


# In[6]:


uncert(id_example)
uncert(ood_example)
uncert(random_example)


# In[ ]:




