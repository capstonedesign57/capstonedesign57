#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('atis_train.csv')
data = data[['tokens', 'slots']]
data = data.dropna()


# In[2]:


for i in range(len(data['tokens'])):
    sent = data['tokens'][i][4:-3]
    tag = data['slots'][i][2:-1]
    data['tokens'][i] = sent
    data['slots'][i] = tag


# In[3]:


tagged_data = pd.DataFrame(columns = ['Sentence #', 'Word', 'Pos', 'Tag'])


# In[4]:


from nltk import word_tokenize, pos_tag


# In[5]:


i = -1
for k in range(len(data['tokens'])) :
    i += 1
    token = word_tokenize(data['tokens'][i])
    poses = [b for (a,b) in pos_tag(token)]
    tags = word_tokenize(data['slots'][i])
    sentence_num = "Sentence "+str(i)
    temp = pd.DataFrame({"Sentence #": sentence_num, "Word": token, "Pos": poses, "Tag": tags})
    tagged_data=pd.concat([tagged_data, temp])


# In[6]:


tagged_data=tagged_data.reset_index(drop=True)


# In[7]:


tagged_data


# In[8]:


tagged_data.to_csv("BIO_TAGGER_TRAINING_DATASET.csv")


# In[ ]:




