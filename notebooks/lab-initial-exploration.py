
# coding: utf-8

# In[ ]:


# automatically show plots inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# reload all modules before executing code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# switch to src directory to avoid import statement confusion
get_ipython().run_line_magic('cd', '../src')

# import statements go here
import seaborn as sns

