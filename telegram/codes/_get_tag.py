#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# To load the model
custom_objects={'CRF': CRF,'crf_loss':crf_loss,'crf_viterbi_accuracy':crf_viterbi_accuracy}
# To load a persisted model that uses the CRF layer 
BIO_TAGGER = load_model('_BIO_TAGGER_1.h5', custom_objects = custom_objects)
BIO_TAGGER._make_predict_function()

# In[5]:


from nltk import word_tokenize


# In[6]:


import pickle

with open('_word_to_index_1.pickle', 'rb') as f1:
    word_to_index = pickle.load(f1)
    
with open('_index_to_tag_1.pickle', 'rb') as f2:
    index_to_tag = pickle.load(f2)    

with open('X_test_1.pickle', 'rb') as f3:
    X_test = pickle.load(f3)    

with open('y_test_1.pickle', 'rb') as f4:
    y_test = pickle.load(f4)    


# In[1]:


import numpy as np
import pandas as pd
from pandas import DataFrame as df
import re
import _todatefomat as todate


# In[2]:


def todf(sent):
    
    tagged = pd.DataFrame(columns=("Word","Prediction"))
    
    sent = sent.lower()
    sent = word_tokenize(sent)
    
    new_X = []
    for w in sent:
        try:
            new_X.append(word_to_index.get(w,1))
        except KeyError:
            new_X.append(word_to_index['OOV'])
            
    max_len = 70
    pad_new = pad_sequences([new_X], padding="post", value=0, maxlen=max_len)
    
    p = BIO_TAGGER.predict(np.array([pad_new[0]]))
    p = np.argmax(p, axis=-1)
    i=0
    for w, pred in zip(sent, p[0]):
        tagged.loc[i]=[w,index_to_tag[pred]]
        i+=1
    return tagged


# In[1]:


def get_element(user_input, intent):  # for first input
    tagged = todf(user_input)
    
    time = re.compile("date|time|day")

    fromloc=''
    stoploc=''
    toloc=''
    arrtime=''
    dpttime=''
    arl=''
    cheapest=0
    cost=''
    
    for i in range(len(tagged['Prediction'])):
        if "below" in user_input and tagged['Word'][i].isdigit():
            cost = tagged['Word'][i]
        if "from" in tagged['Prediction'][i]: #from
            fromloc = fromloc+' '+tagged['Word'][i]
            fromloc = fromloc.lstrip()
        elif "stop" in tagged['Prediction'][i]: #stop
            stoploc = stoploc+' '+tagged['Word'][i]
            stoploc = stoploc.lstrip()
        elif "to" in tagged['Prediction'][i]:  # default로 도착지
            toloc = toloc+' '+tagged['Word'][i]   
            toloc = toloc.lstrip()
        elif time.search(tagged['Prediction'][i]): # date, time, day
            date=todate.date(tagged)
            if "arrive" in tagged['Prediction'][i] or "return" in tagged['Prediction'][i]: # arrive, return
                arrtime = date
            else:  # default로 출발시간
                dpttime = date
        elif "airline" in tagged['Prediction'][i] : # airline
            if tagged['Word'][i]!='below' and tagged['Word'][i].isdigit()==False:
                arl = arl+' '+tagged['Word'][i]
                arl = arl.lstrip()
        elif "cost" in tagged['Prediction'][i]: # cost 
            cheapest = 1
        elif "fare" in tagged['Prediction'][i]: # fare
            cost = cost+' '+tagged['Word'][i]
            cost = cost.lstrip()
        else: 
            continue
            
    result = df(data={'tag':['airline','fromloc','stoploc','toloc','dpttime','arrtime','cost','cheapest'],
                 'element':[arl,fromloc,stoploc,toloc,dpttime,arrtime,cost,cheapest]}, columns=['tag','element'])
    
    #의도는 있는데 요소가 없을 경우
    if 'AskFlightWithAirline' in intent:
        if result[result['tag'].isin(['airline'])].empty==None:
            print('Which airlilne would you like')
            #사용자에게 input을 받아온다
            result.element[0]=input()
    if 'AskFlightWithCost' in intent:
        if result[result['tag'].isin(['cheapest'])].element.tolist()[0]==0 and result[result['tag'].isin(['cost'])].empty==False:
            print("Do you want the cheapest flight or a ticket below a certain price?")
            reply=input()
            if "cheapest" in reply:
                result.element[7]=1
            else:
                if reply.isdigit:
                    result.element[6]='\''+re.findall('\d+', reply)[0]+'\''
                else:
                    print("Sorry, I didn't get that. Below how much?")
                    reply=input()
                    result.element[6]='\''+re.findall('\d+', reply)[0]+'\''
    
    return result


# In[2]:


def ellipsis(result, user_input, second_input):
    flag = 0
    df = todf(second_input)
    
    loc = re.compile("airport|city|country|state")
    time = re.compile("date|time|day")
    fromloc=''
    stoploc=''
    toloc=''
    arl=''
    new_sent=user_input
    
    for w in df['Prediction']:
        if "below" in second_input and re.findall('\d+', df[df['Prediction'].isin([w])].Word.tolist()[0]) and flag==0:
            if result.element[6]=='':
                result.element[6] = df[df['Prediction'].isin([w])].Word.tolist()[0]
                new_sent = new_sent+" below "+result.element[6]
                flag=1
            result = result.replace(result.element[6], df[df['Prediction'].isin([w])].Word.tolist()[0])
            result.element[7]=''
        elif "from" in w: #from
            fromloc = fromloc+' '+df[df['Prediction'].isin([w])].Word.tolist()[0]
            fromloc = fromloc.lstrip()
            result = result.replace(result.element[1], fromloc)
        elif "stop" in w: #stop
            stoploc = stoploc+' '+df[df['Prediction'].isin([w])].Word.tolist()[0]
            stoploc = stoploc.lstrip()
            result = result.replace(result.element[2], stoploc)
        elif "to" in w:  # default로 도착지
            toloc = toloc+' '+df[df['Prediction'].isin([w])].Word.tolist()[0] 
            toloc = toloc.lstrip()
            result = result.replace(result.element[3], toloc)
        elif loc.search(w):
            if "from" in second_input: #from
                fromloc = fromloc+' '+df[df['Prediction'].isin([w])].Word.tolist()[0]
                fromloc = fromloc.lstrip()
                result = result.replace(result.element[1], fromloc)
            elif "stop" in second_input: #stop
                stoploc = stoploc+' '+df[df['Prediction'].isin([w])].Word.tolist()[0]
                stoploc = stoploc.lstrip()
                result = result.replace(result.element[2], stoploc)
            else:  # default로 도착지
                toloc = toloc+' '+df[df['Prediction'].isin([w])].Word.tolist()[0] 
                toloc = toloc.lstrip()
                result = result.replace(result.element[3], toloc)
        elif time.search(w): # date, time, day
            date=todate.date(df)
            if "arrive" in second_input or "return" in second_input: # arrive, return
                if date[:7]==result.element[5][:7]:
                    date=result.element[5][:7]+date[-3:]
                elif date[:3]==result.element[5][:3]:
                    date=result.element[5][:5]+date[-5:]
                else: 
                    continue
                result = result.replace(result.element[5], date)
            else:  # default로 출발시간
                if date[:7]==result.element[4][:7]:
                    date=result.element[4][:7]+date[-3:]
                elif date[:3]==result.element[4][:3]:
                    date=result.element[4][:5]+date[-5:]
                else: 
                    continue
                result = result.replace(result.element[4], date)
        elif "airline" in w : # airline
            if result.element[0]=='':
                result.element[0] = df[df['Prediction'].isin([w])].Word.tolist()[0]
                new_sent = new_sent+" by "+result.element[0]
            arl = arl+' '+df[df['Prediction'].isin([w])].Word.tolist()[0]
            arl = arl.lstrip()
            new_sent = new_sent.replace(result.element[0], arl)
            result = result.replace(result.element[0], arl)
        elif "cost" in w: # cost 
            if result.element[7]=='':
                result.element[7] = df[df['Prediction'].isin([w])].Word.tolist()[0]
                new_sent = new_sent+", the cheapest one"
            result = result.replace(result.element[7], df[df['Prediction'].isin([w])].Word.tolist()[0])
        elif "fare" in w: # fare
            if result.element[6]=='':
                result.element[6] = df[df['Prediction'].isin([w])].Word.tolist()[0]
                new_sent = new_sent+" below "+result.element[6]
            result = result.replace(result.element[6], df[df['Prediction'].isin([w])].Word.tolist()[0])
        else:
            continue
            
    return new_sent, result