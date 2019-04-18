#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from data_utils import *
from plot_utils import *

from midnite.visualization import graduam
from midnite.common import Flatten
import midnite
from midnite.uncertainty import *


# In[2]:


alexnet = models.alexnet(pretrained=True)
alexnet.classifier.add_module("softmax", Softmax(dim=1))


# In[3]:


id_example = get_example_from_path("../data/imagenet_example_283.jpg", DataConfig.ALEX_NET)
ood_example = get_example_from_path("../data/ood_example.jpg", DataConfig.ALEX_NET)


# In[4]:


random_example = get_random_example(DataConfig.ALEX_NET)


# In[5]:


def uncert(img):
    with midnite.device("cpu"): # cuda:0 if available
        result = graduam(StochasticDropouts(Sequential(*(list(alexnet.features.children()) + [alexnet.avgpool]))),
                         StochasticDropouts(Sequential(*([Flatten()] + list(alexnet.classifier.children())))),
                        img, 50)
        show_heatmap(result, 1.5, img)


# In[6]:


uncert(id_example)
uncert(ood_example)
uncert(random_example)

