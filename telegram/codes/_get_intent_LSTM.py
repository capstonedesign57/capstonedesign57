#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from keras.models import load_model
import keras


# In[2]:


from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from nltk import sent_tokenize, word_tokenize
from keras import backend as k

# wordvector load
from gensim.models import FastText
model = FastText.load('fasttext_model')
fasttext = model.wv


# In[3]:


def pred1(model, sentence) :
    _dtype = k.floatx()
    sentence_token = []
    sentence_token += word_tokenize(sentence)
    
    sentence_vec = []
    sentence_vec.append([fasttext[v] for v in sentence_token])
    
    padd = sequence.pad_sequences(sentence_vec, maxlen = 45, dtype = _dtype)
    intent = ans1(model.predict(padd)[0])
    
    return(intent)


# In[4]:


def ans1(lst):
    ans = ["Flight", "NoFlight"]
    temp = [[i, lst[i]] for i in range(2)]
    temp.sort(key = lambda x : x[1], reverse = True)

    return ans[temp[0][0]]


# In[5]:


def pred2(model, sentence) :
    sentence_token = []
    sentence_token += word_tokenize(sentence)
    
    sentence_vec = []
    sentence_vec.append([fasttext[v] for v in sentence_token])
    _dtype = k.floatx()
    padd = sequence.pad_sequences(sentence_vec, maxlen = 45, dtype = _dtype)
    intent = ans2(model.predict(padd)[0])
    
    return intent


# In[6]:


def ans2(lst) :
    ans = ["AskFlight", "AskFlight, AskFlightWithCost", 
           "AskFlight, AskFlightWithAirline", "AskFlight, AskFlightWithCost, AskFlightWithAirline"]
    temp = [[i, lst[i]] for i in range(4)]
    temp.sort(key = lambda x : x[1], reverse = True)

    return ans[temp[0][0]]


# In[7]:


def get_intent(sent):
    # isflight
    with keras.backend.get_session().graph.as_default():
        LSTM_isflight = load_model('LSTM_model_1.h5')
        predict = pred1(LSTM_isflight, sent)

    if predict == 'NoFlight':
        return False

    #항공권 검색 질의일 경우
    # intent 4가지(flight, cost, airline, cost+airline)
    with keras.backend.get_session().graph.as_default():
        LSTM_intent4 = load_model('LSTM_model.h5')
        predict = pred2(LSTM_intent4, sent)
    
    return predict


# In[8]:


get_intent("i want a flight from incheon to jeju by korean air")


# In[9]:


get_intent("what time is it")


# In[ ]:




