
# coding: utf-8

# In[1]:


import pandas as pd
import nltk, re
from nltk import sent_tokenize, word_tokenize

csv_test = pd.read_csv("C:/Users/조혜령/Desktop/졸업프로젝트/data/atis.csv")
csv_test = csv_test[['sentences','intend']]
csv_test = csv_test.dropna()
# csv_test.tail()


# In[2]:


raw_sents = csv_test['sentences']
#raw_sents.head()
tokenized_sent = []
tokens = []
for line in raw_sents:
    tokenized_sent.append(word_tokenize(line))
    tokens += word_tokenize(line)


# In[3]:


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


# # 미리 학습된 모델과 병합

# In[4]:


# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
# URL에서 미리 학습된 모델을 다운로드 받는다.
file_name = 'D:\\졸업프로젝트\GoogleNews-vectors-negative300.bin'
model.intersect_word2vec_format(fname=file_name, binary=True)


# # 최종 모델을 저장한다

# In[5]:


model.save('word2vec.model')


# # 저장한 모델을 읽어서 이용한다

# In[6]:


model = Word2Vec.load('word2vec.model')
word_vectors = model.wv


# In[7]:


vector_sent = []
for sent in tokenized_sent:
    vector_sent.append([word_vectors[v] for v in sent])
len(vector_sent)


# In[8]:


import numpy as np
import y
intend = csv_test['intend']
vector_intend = []
for i in intend : 
    vector_intend.append(y.ans(i))
#vector_intend = np.array(vector_intend)
#vector_intend = [w.reshape(1,11) for w in vector_intend] 
#vector_intend


# # RNN model

# In[9]:


import tensorflow as tf


# In[10]:


batch_size = 1
h_size = 1 
w_size = 300 #워드 벡터 차원수 (1(h_size) x 300 (w_size))
c_size = 1
hidden_size = 100 
out_size = 11

x_raw = tf.placeholder(tf.float32, shape = [batch_size, 17, w_size])
x_split = tf.split(x_raw, 17, axis=1)
ans = tf.placeholder(tf.float32, shape = [batch_size, out_size])
U = tf.Variable(tf.random_normal([w_size, hidden_size], stddev=0.01))
W = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01))
V = tf.Variable(tf.random_normal([hidden_size, out_size], stddev=0.01))

s={}
s_init = tf. random_normal(shape=[batch_size, hidden_size], stddev=0.01)
s[-1] = s_init
x_split[0].shape


# In[11]:


i=0
for t, word_vec in enumerate(x_split): 
    xt = tf.reshape(word_vec, [batch_size, w_size])
    s[t] = tf.nn.tanh(tf.matmul(xt, U) + tf.matmul(s[t-1], W))
    i += 1

o = tf.nn.softmax(tf.matmul(s[i-1], V))

cost = -tf.reduce_mean(tf.log(tf.reduce_sum(o*ans, axis=1)))


# # Training

# In[12]:


learning_rate = 0.1
trainer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# In[13]:


sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
init.run()


# # 모델의 정확도를 계산하기 위한 함수

# In[14]:


def accuracy(network, t):
    
    t_predict = tf.argmax(network, axis=1)
    t_actual = tf.argmax(t, axis=1)

    return tf.reduce_mean(tf.cast(tf.equal(t_predict, t_actual), tf.float32))


# # 실제로 학습이 수행되는 부분

# # 모델의 정확도를 평가해보자

# In[16]:


acc = accuracy(o,ans)
acc.eval({x_raw: vector_sent[0:1], ans: vector_intend[0:1]})

