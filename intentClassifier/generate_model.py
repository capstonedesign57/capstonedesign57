#!/usr/bin/env python
# coding: utf-8

# # FastText model 만들기

# In[1]:


from gensim.models import FastText


# In[ ]:


# wiki.en.bin 다운 경로: https://fasttext.cc/docs/en/crawl-vectors.html


# In[2]:


model = FastText.load_fasttext_format('./wiki.en.bin')


# In[3]:


model.save('fasttext_model')


# In[4]:


print(model.most_similar('incheon'))


# In[5]:


print(model.wv.most_similar('incheon'))


# In[20]:


print(model.most_similar('teacher'))
print(model.similarity('flies', 'fly'))
print(model.most_similar('flight'))
print(model.most_similar('incheon'))

