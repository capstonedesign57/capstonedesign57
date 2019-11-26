#!/usr/bin/env python
# coding: utf-8

# # LSTM_Flight_NoFlight

# In[1]:


from gensim.models import FastText

model = FastText.load('fasttext_model')
fasttext = model.wv


# In[2]:


import numpy as np
import pandas as pd

data = pd.read_csv('lstm_train_flight_noflight_1.csv')
data = data[['tokens', 'intent']]

X_data = data['tokens']
y_data = data['intent']


# In[3]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

pd.value_counts(data['intent']).plot.bar()
plt.title('histogram')
plt.xlabel('intent')
plt.ylabel('Frequency')

data['intent'].value_counts()


# In[4]:


from nltk import sent_tokenize, word_tokenize
import bool_intent

tokenized_X = [] # 토큰화한 문장 (단어, 단어, 단어, ...)
for line in X_data :
    tokenized_X.append(word_tokenize(line))

vectored_X = [] # 토큰화한 문장을 벡터로
for sent in tokenized_X :
    vectored_X.append([fasttext[v].tolist() for v in sent])
    
vectored_y = []
for index in y_data : 
    vectored_y.append(bool_intent.intent(index))
    
X = np.array(vectored_X)
y = np.array(vectored_y)

print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))


# In[5]:


from keras.preprocessing.sequence import pad_sequences
from keras import backend as k

_dtype = k.floatx()
max_len = max(len(l) for l in X) # token의 최대 길이
data = pad_sequences(X, maxlen = max_len, dtype = _dtype)


# In[6]:


n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)


# In[7]:


X_train = data[ : n_of_train]
y_train = y[ : n_of_train]
X_test = data[n_of_train : ]
y_test = y[n_of_train : ]


# # RNN 학습시키는 부분

# In[8]:


from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential

model = Sequential()
model.add(LSTM(30, activation='tanh'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 42, epochs = 20)
# batch_size는 한 번에 학습하는 데이터의 개수

print(model.summary())


# In[9]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_true = y_test
y_pred = model.predict_classes(X_test, verbose = 0)

print(classification_report(y_true, y_pred, target_names = ['Flight', 'NoFlight']))


# In[10]:


def ans(lst):
    ans = ["Flight", "NoFlight"]
    temp = [[i, lst[i]] for i in range(2)]
    temp.sort(key = lambda x : x[1], reverse = True)

    print(ans[temp[0][0]])


# In[11]:


from keras.preprocessing import sequence
from keras import backend as k
_dtype = k.floatx()

def predict(sentence) :
    sentence = sentence
    sentence_token = []
    sentence_token += word_tokenize(sentence)
    
    sentence_vec = []
    sentence_vec.append([fasttext[v] for v in sentence_token])
    
    padd = sequence.pad_sequences(sentence_vec, maxlen = max_len, dtype = _dtype)
    intent = ans(model.predict(padd)[0])
    
    return(intent)


# In[12]:


# model 저장하기

model_json = model.to_json()
with open("LSTM_model_1.json", "w") as json_file :
    json_file.write(model_json)
    
model.save_weights("LSTM_model_1.h5")


# In[13]:


# 임의 지정한 문장으로 의도 예측 테스트

predict('new york arriving denver') # Flight
predict('what type of aircraft') # NoFlight
predict('all airport') # Noflight
predict('from new york to denver one way fare') # Flight
predict('what ground transportation is available in denver') # NoFlight
predict('ewha') # NoFlight

