#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


def pred(model, sentence) :
    sentence_token = []
    sentence_token += word_tokenize(sentence)
    
    sentence_vec = []
    sentence_vec.append([fasttext[v] for v in sentence_token])
    _dtype = k.floatx()
    padd = sequence.pad_sequences(sentence_vec, maxlen = 42, dtype = _dtype)
    intent = ans(model.predict(padd)[0])
    
    return intent


# In[4]:


def ans(lst) :
    ans = ["AskFlight", "AskFlight, AskFlightWithCost", 
           "AskFlight, AskFlightWithAirline", "AskFlight, AskFlightWithCost, AskFlightWithAirline"]
    temp = [[i, lst[i]] for i in range(4)]
    temp.sort(key = lambda x : x[1], reverse = True)

    return ans[temp[0][0]]


# In[1]:


def get_intent(sent):
    # isflight
    filename = 'MNB_model_1.sav'
    MNB_isflight = pickle.load(open(filename, 'rb'))
    dtmvector = pickle.load(open('dtmvector_1', 'rb'))
    tfidf_transformer = pickle.load(open('tfidf_transformer_1', 'rb'))
    
    test = []
    test.append(sent)
    test_dtm = dtmvector.transform(test)
    tfidfv_test = tfidf_transformer.transform(test_dtm)
    predict = MNB_isflight.predict(tfidfv_test)
    
    # 항공권 검색 질의가 아닐 경우
    if predict=='NoFlight':
        return False
    
    #항공권 검색 질의일 경우
    # intent 4가지(flight, cost, airline, cost+airline)
    with keras.backend.get_session().graph.as_default():
        LSTM_intent4 = load_model('LSTM_model.h5')
        predict = pred(LSTM_intent4, sent)
    
    return predict


# In[7]:


get_intent("i want a flight from incheon to jeju by korean air")


# In[ ]:




