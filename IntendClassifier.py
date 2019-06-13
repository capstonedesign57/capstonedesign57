#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk, re
from nltk import sent_tokenize, word_tokenize

train_data = pd.read_csv("atis.csv")
train_data = train_data[['sentences','intend']]
train_data = train_data.dropna()


# In[4]:


raw_sents = train_data['sentences']
tokenized_sents = []
tokens = []
for line in raw_sents:
    tokenized_sents.append(word_tokenize(line))
    tokens += word_tokenize(line)


# In[5]:


import numpy as np
import Untitled4
intend = train_data['intend']
vector_intend = []
for i in intend : 
    vector_intend.append(Untitled4.ans(i))


# In[6]:


import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
model = gensim.models.Word2Vec(
    tokenized_sent,
    size = 300,
    window = 10,
    min_count = 1,
    workers = 10,
    sample = 1e-3,
    sg = 1)
model.init_sims(replace=True)
model.save('atis')


# In[7]:


# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
# URL���� �̸� �н��� ���� �ٿ�ε� �޴´�.
file_name = 'C:/Users/������/Desktop/����������Ʈ/ex/GoogleNews-vectors-negative300.bin'
model.intersect_word2vec_format(fname=file_name, binary=True)


# In[8]:


model.save('word2vec.model')


# In[9]:


model = Word2Vec.load('word2vec.model')
word_vectors = model.wv


# In[10]:


vector_sent = []
for sent in tokenized_sents:
    vector_sent.append([word_vectors[v] for v in sent])


# In[11]:


import tensorflow as tf


# In[12]:


from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
_num_timesteps = 3  #�ð迭 time step ��
_dtype = K.floatx()  # backend engine float type

maxlen = 25
padded = pad_sequences(vector_sent, maxlen, dtype = _dtype, padding='post', value = -1)


# In[13]:


batch_size = 1
h_size = maxlen
w_size = 300 #���� ���� ������ (1(h_size) x 300 (w_size))
c_size = 1
hidden_size = 100 
out_size = 11


# In[14]:


x_raw = tf.placeholder(tf.float32, shape = [batch_size, h_size, w_size])

x_split = tf.split(x_raw, h_size, axis=1)
ans = tf.placeholder(tf.float32, shape = [batch_size, out_size])
U = tf.Variable(tf.random_normal([w_size, hidden_size], stddev=0.01))
W = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01))
V = tf.Variable(tf.random_normal([hidden_size, out_size], stddev=0.01))

s={}
s_init = tf. random_normal(shape=[batch_size, hidden_size], stddev=0.01)
s[-1] = s_init


# In[15]:


i=0
for t, word_vec in enumerate(x_split): 
    xt = tf.reshape(word_vec, [batch_size, w_size])
    s[t] = tf.nn.tanh(tf.matmul(xt, U) + tf.matmul(s[t-1], W))
    i += 1

o = tf.nn.softmax(tf.matmul(s[i-1], V))

cost = -tf.reduce_mean(tf.log(tf.reduce_sum(o*ans, axis=1)))


# In[16]:


learning_rate = 0.1
trainer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[17]:


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
init.run()


# In[18]:


for _ in range(20):
    for i in range(len(vector_sent)):
        inputs = padded[i:i+1]  # ���⼭�� padded�� -1���� �߶󳻾�
        outputs = vector_intend[i:i+1]
        feed = {x_raw:inputs, ans : outputs}
        trainer.run(feed)


# In[19]:


def accuracy(network, t):
    
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))


# In[20]:


acc = accuracy(o,ans)
for i in range (len(vector_sent)):
    print(acc.eval({x_raw: padded[i:i+1], ans: vector_intend[i:i+1]}))

