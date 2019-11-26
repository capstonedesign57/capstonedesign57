#!/usr/bin/env python
# coding: utf-8

# # Intent_4

# In[14]:


import nltk
from nltk import word_tokenize

"""def intent(lst):
    y = [0 for i in range(1)]
    for element in word_tokenize(lst) :
        if element.lower() == "askflight" :
            y[0] = 0
        elif element.lower() == "askflight, askflightwithcost" :
            y[0] = 1
        elif element.lower() == "askflight, askflightwithairline" :
            y[0] = 2
        else : # element.lower() == "askflight, askflightwithcost, askflightwithairline"
            y[0] = 3
    return y"""

def intent(lst):
    if lst.lower() == "askflight" :
        y = 0
    elif lst.lower() == "askflight, askflightwithcost" :
        y = 1
    elif lst.lower() == "askflight, askflightwithairline" :
        y = 2
    else : # element.lower() in "askflight, askflightwithcost, askflightwithairline"
        y = 3
    return y

