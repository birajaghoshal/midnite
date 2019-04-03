#!/usr/bin/env python
# coding: utf-8

# In[45]:


import torch

a = torch.ones((3, 4, 4))
b = torch.tensor([1, 2, 3]).float().unsqueeze(dim=1).unsqueeze(dim=2)
#b = torch.tensor([[[1]], [[2]], [[3]]]).float()

a * b


# In[59]:


a = torch.ones((3, 4, 2))
b = torch.ones((4, 2))

b * a


# In[54]:


a = torch.ones(3, 5, 5)
b = torch.tensor([1]).float()
b * a
a * b


# In[ ]:




