#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv('BIO_TAGGER_TRAINING_DATASET_1_TIME_1.csv')


# In[2]:


import nltk


# In[3]:


del data['Pos']
del data['Unnamed: 0']


# In[4]:


data


# In[5]:


SentList=list(data["Sentence #"])
TagList=list(data["Tag"])


# In[6]:


print(len(SentList))
print(len(TagList))


# In[7]:


i=0
temp=[]
for i in range(len(SentList)) :
    if "arrive" in TagList[i]:
        temp.append(SentList[i])
        #temp.append(SentList[i][9:])


# In[8]:


arriveSet=set(temp)


# In[9]:


len(arriveSet)


# In[10]:


i=0
temp=[]
for i in range(len(SentList)) :
    if "depart" in TagList[i]:
        temp.append(SentList[i])
        #temp.append(SentList[i][9:])


# In[11]:


departSet=set(temp)


# In[12]:


len(departSet)


# In[13]:


departSet=list(departSet)


# In[14]:


departDF=pd.DataFrame(data=departSet, columns=['sent#'])


# In[15]:


departDF


# In[16]:


departSample=departDF.sample(frac=0.18, random_state=777)


# In[17]:


departSample = list(departSample['sent#'])


# In[18]:


departSample.sort()


# In[19]:


len(departSample)


# In[20]:


departSample


# In[21]:


arriveSet=list(arriveSet)


# In[22]:


time=arriveSet+departSample


# In[23]:


len(set(time))


# In[24]:


Data_Time = pd.DataFrame(columns=['Sentence #', 'Word','Tag'])


# In[25]:


i=0
j=0
for i in range(len(data)):
    if data.iloc[i]['Sentence #'] in time :
        Data_Time.loc[j]=data.loc[i]
        j+=1


# In[29]:


len(Data_Time['Sentence #'])


# In[27]:


Data_Time.to_csv("BIO_TAGGER_TRAINING_DATASET_1_TIME.csv",sep=',')


# In[ ]:




