#!/usr/bin/env python
# coding: utf-8

# # Flight/NoFlight Intent Classifier

# In[1]:


import nltk
from nltk import word_tokenize

"""def intent(lst) :
    y = [0 for index in range(2)]
    for element in word_tokenize(lst) :
        if element.lower() == "Flight" :
            y[0] = 1
        else :
            y[1] = 1
    return y"""

def intent(lst) :
    y = [0 for index in range(1)]
    for element in word_tokenize(lst) :
        if element.lower() == "flight" :
            y[0] = 0
        elif element.lower() == "noflight" :
            y[0] = 1
    return y

