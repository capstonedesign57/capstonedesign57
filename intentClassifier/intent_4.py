#!/usr/bin/env python
# coding: utf-8

# # Intent_4

# In[1]:


import nltk
from nltk import word_tokenize


# In[2]:


def intent(lst):
    if lst.lower() == "askflight" :
        y = 0
    elif lst.lower() == "askflight, askflightwithcost" :
        y = 1
    elif lst.lower() == "askflight, askflightwithairline" :
        y = 2
    else : # lst.lower() in "askflight, askflightwithcost, askflightwithairline"
        y = 3
    return y

