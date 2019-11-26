#!/usr/bin/env python
# coding: utf-8

# # LSTM_4_Intent

# In[1]:


from gensim.models import FastText

model = FastText.load('fasttext_model')
fasttext = model.wv


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_4_intent_1.csv')
data = data[['tokens', 'intent']]

X_data = data['tokens']
y_data = data['intent']


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

pd.value_counts(data['intent']).plot.bar()
plt.title('histogram')
plt.xlabel('intent')
plt.ylabel('Frequency')

data['intent'].value_counts()


# In[4]:


from nltk import sent_tokenize, word_tokenize
import import_ipynb
import intent_4 # AskFlight, AskFlight/AskFlightWithCost, AskFlight/AskFlightWithAirline
                # AskFlight, AskFlightWithCost, AskFlightWithAirline(AskAll)

tokenized_X = [] # 토큰화한 문장 (단어, 단어, 단어, ...)
for line in X_data :
    tokenized_X.append(word_tokenize(line))

vectored_X = [] # 토큰화한 문장을 벡터로
for sent in tokenized_X :
    vectored_X.append([fasttext[v].tolist() for v in sent])

vectored_y = []
for index in y_data : 
    vectored_y.append(intent_4.intent(index))


# In[5]:


X = np.array(vectored_X)
y = np.array(vectored_y)

print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))


# In[6]:


from keras.preprocessing.sequence import pad_sequences
from keras import backend as k

_dtype = k.floatx()
max_len = max(len(l) for l in X) # token의 최대 길이
data = pad_sequences(X, maxlen = max_len, dtype = _dtype)


# In[7]:


max_len


# In[8]:


n_of_train = int(len(X_data) * 0.85)
n_of_test = int(len(X_data) - n_of_train)


# In[9]:


X_train = data[ : n_of_train]
y_train = y[ : n_of_train]
X_test = data[n_of_train : ]
y_test = y[n_of_train : ]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# # RNN 학습시키는 부분

# In[10]:


from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential

model = Sequential()
model.add(LSTM(30, activation='tanh'))
model.add(Dense(4, activation = 'softmax')) # 출력층
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 42, epochs = 50)
# batch_size는 한 번에 학습하는 데이터의 개수

print(model.summary())


# In[11]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_true = y_test
y_pred = model.predict_classes(X_test, verbose = 0)

print(classification_report(y_true, y_pred, 
                            target_names = ['AskFlight', 'AskFlight, AskFlightWithCost',
                                            'AskFlight, AskFlightWithAirline', 
                                            'AskFlight, AskFlightWithCost, AskFlightWithAirline']))


# In[12]:


def ans(lst) :
    ans = ["AskFlight", "AskFlight, AskFlightWithCost", 
           "AskFlight, AskFlightWithAirline", "AskFlight, AskFlightWithCost, AskFlightWithAirline"]
    temp = [[i, lst[i]] for i in range(4)]
    temp.sort(key = lambda x : x[1], reverse = True)

    print(ans[temp[0][0]])


# In[13]:


from keras.preprocessing import sequence

def predict(sentence) :
    sentence = sentence
    sentence_token = []
    sentence_token += word_tokenize(sentence)
    
    sentence_vec = []
    sentence_vec.append([fasttext[v] for v in sentence_token])
    
    padd = sequence.pad_sequences(sentence_vec, maxlen = max_len, dtype = _dtype)
    intent = ans(model.predict(padd)[0])
    
    return(intent)


# In[1]:


from keras.models import *
from keras.utils import *

model.save('LSTM_model.h5')


# In[ ]:


# 임의 지정한 문장으로 의도 예측 테스트

predict('from la to san francisco') # AskFlight
predict('cheapest one from miami to denver') # AskFlight, AskFlightWithCost
predict('cheapest from denver to philadelphia on monday by delta') # AskFlight, AskFlightWithCost, AskFlightWithAirline
predict('denver to philadelphia under 100 dollars') # AskFlight, AskFlightWithCost
predict('flight from miami to denver on monday') # AskFlight
predict('one way fare from miami to denver on delta') # AskFlight, AskFlightWithCost, AskFlightWithAirline
predict('flight from incheon to jeju on monday under 50 dollars') # AskFlight, AskFlightWithCost


# In[ ]:




