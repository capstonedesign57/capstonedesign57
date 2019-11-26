#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model
# To load the model
custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
# To load a persisted model that uses the CRF layer 
BIO_TAGGER = load_model('_BIO_TAGGER.h5', custom_objects = custom_objects)


# In[2]:


from nltk import word_tokenize


# In[3]:


import pickle

with open('_word_to_index.pickle', 'rb') as f1:
    word_to_index = pickle.load(f1)
    
with open('_index_to_tag.pickle', 'rb') as f2:
    index_to_tag = pickle.load(f2)    

with open('X_test.pickle', 'rb') as f3:
    X_test = pickle.load(f3)    

with open('y_test.pickle', 'rb') as f4:
    y_test = pickle.load(f4) 

with open('_index_to_word.pickle', 'rb') as f5:
    index_to_word = pickle.load(f5)


# In[4]:


from keras.utils import np_utils
y_test2 =  np_utils.to_categorical(y_test)


# In[5]:


import numpy as np


# In[6]:


i=3
p=BIO_TAGGER.predict(np.array([X_test[i]]))
p=np.argmax(p, axis=-1)
true = np.argmax(y_test2[i],-1)

for w, t, pred in zip(X_test[i],true, p[0]):
    if t != 0: # PAD값은 제외함.
        print("{:17}: {:30} {}".format(index_to_word[w], index_to_tag[t], index_to_tag[pred]))


# In[7]:


true


# In[8]:


for w, t, pred in zip(X_test[i],true, p[0]):
    if t != 0: # PAD값은 제외함.
        print("{:17}: {:30} {}".format(index_to_word[w], index_to_tag[t], t))


# In[9]:


#from keras.utils import np_utils
y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (BIO_TAGGER.evaluate(X_test, y_test2)[1]))


# In[10]:


Check = []
for i in range(len(X_test)) : 
    Check.append([])
    p=BIO_TAGGER.predict(np.array([X_test[i]]))
    p=np.argmax(p, axis=-1)
    true = np.argmax(y_test2[i],-1)
    for j in range(len(true)) :
        if true[j] != 0:
            if p[0][j] == true[j] :
                Check[i].append(1)
            else :
                Check[i].append(0)


# In[11]:


true


# In[12]:


Ratio_right = []
Ratio_wrong = []


# In[13]:


for C in Check :
    count=0
    for r in C :
        if r == 1 :
            count += 1
    ratio = count/len(C)
    Ratio_right.append(ratio)
    Ratio_wrong.append(1-ratio)


# In[14]:


len(C)


# In[15]:


from numpy import array
from matplotlib import pyplot

data1 = array(Ratio_right)
data2 = array(Ratio_wrong)
pyplot.bar(range(len(data1)), data1, color = "#0000FF")
pyplot.bar(range(len(data2)), data2, bottom=data1, color = "#FF0000")


# In[16]:


avg2 = np.mean(data1)
print(avg2)


# In[17]:


Check_1 = []
for i in range(len(X_test)) : 
    Check_1.append([])
    p=BIO_TAGGER.predict(np.array([X_test[i]]))
    p=np.argmax(p, axis=-1)
    true = np.argmax(y_test2[i],-1)
    for j in range(len(true)) :
        if true[j] != 0 and true[j] != 22:   # 첫번째 조건은 패딩 부분 제외, 두번째 조건은 O가 아닌 것들만
            if p[0][j] == true[j] :
                Check_1[i].append(1)
            else :
                Check_1[i].append(0)


# In[18]:


Ratio_right_1 = []
Ratio_wrong_1 = []


# In[19]:


for I in range(len(Check_1)) :
    if len(Check_1[i]) == 0 :
        print(i)


# In[24]:


Check_1


# In[20]:


for C in Check_1 :
    count_1=0
    for r in C :
        if r == 1 :
            count_1 += 1
    if len(C)!=0:
        ratio_1 = count_1/len(C)
        Ratio_right_1.append(ratio_1)
        Ratio_wrong_1.append(1-ratio_1)


# In[21]:


len(C)


# In[22]:


from numpy import array
from matplotlib import pyplot

data1_1 = array(Ratio_right_1)
data2_1 = array(Ratio_wrong_1)
pyplot.bar(range(len(data1_1)), data1_1, color = "#0000FF")
pyplot.bar(range(len(data2_1)), data2_1, bottom=data1_1, color = "#FF0000")


# In[23]:


avg = np.mean(data1_1)
print(avg)


# In[ ]:




