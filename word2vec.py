
# coding: utf-8

# # Word2Vec
# https://radimrehurek.com/gensim/models/word2vec.html

# In[1]:


import nltk, re
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
f = open('C://Users/조혜령/Desktop/졸업프로젝트/data/atis.txt', 'r', encoding='UTF8') 
raw = f.readlines()
sent = []
for line in raw:
    sent.append(word_tokenize(line)[2:])
clean_sent = []
for line in sent:
    clean_sent.append([t for t in line if (t not in stop)])

import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
model = gensim.models.Word2Vec(
    clean_sent,
    size = 300,
    window = 10,
    min_count = 1,
    workers = 10,
    sample = 1e-3,
    sg = 1)
model.init_sims(replace=True)
model.save('atis')


# # 미리 학습된 모델과 병합

# In[2]:


# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
# URL에서 미리 학습된 모델을 다운로드 받는다.
file_name = 'D:\\졸업프로젝트\GoogleNews-vectors-negative300.bin'
model.intersect_word2vec_format(fname=file_name, binary=True)


# # 최종 모델을 저장한다.

# In[3]:


model.save('word2vec.model')


# # 저장한 모델을 읽어서 이용한다.

# In[4]:


model = Word2Vec.load('word2vec.model')
word_vectors = model.wv


# In[5]:


from nltk import pos_tag
tagged_tokens=[]
for line in sent:
    tagged_tokens+=pos_tag(line)


# In[6]:


clean_tagged_tokens=[]
clean_tagged_tokens+=[(w,pos) for (w,pos) in tagged_tokens if w not in stop]


# In[7]:


verb_tokens=[]
verb_tokens+=[w for (w,pos) in clean_tagged_tokens if pos in ['VB','VBZ','VBD','VBP','VBN','VBG']]


# In[8]:


noun_tokens=[]
noun_tokens+=[w for (w,pos) in clean_tagged_tokens if pos in ['NN','NNS']]


# In[10]:


verb_vectors_list = [word_vectors[v] for v in verb_tokens]
noun_vectors_list = [word_vectors[v] for v in noun_tokens]


# In[11]:


import matplotlib.pyplot as plt
def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8 ,6))
    plt.scatter(xs, ys, marker = 'o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))


# In[13]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
verb_xys = pca.fit_transform(verb_vectors_list)
verb_xs = verb_xys[:,0]
verb_ys = verb_xys[:,1]
plot_2d_graph(verb_tokens, verb_xs, verb_ys)


# In[14]:


noun_xys = pca.fit_transform(noun_vectors_list)
noun_xs = noun_xys[:,0]
noun_ys = noun_xys[:,1]
plot_2d_graph(noun_tokens, noun_xs, noun_ys)

