#!/usr/bin/env python
# coding: utf-8

# # MNB_Flight_NoFlight

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train_flight_noflight.csv')
data = data[['tokens', 'intent']]
data = data.dropna()


# In[2]:


label = data['intent']


# In[3]:


flightcount, noflightcount=0,0
for y in label:
    if y=='Flight':
        flightcount = flightcount+1
    else:
        noflightcount=noflightcount+1
        
print(flightcount, noflightcount, flightcount+noflightcount)


# In[4]:


n_of_train = int(len(data['tokens']) * 0.8)
n_of_test = int(len(data['tokens']) - n_of_train)


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer # 입력한 텍스트를 자동으로 BoW로 만듦

train_data = data['tokens'][0 : n_of_train]
train_data_intent = data['intent'][0 : n_of_train]

dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(train_data)


# In[6]:


from sklearn.feature_extraction.text import TfidfTransformer # TF-IDF를 자동 계산해주는 TfidVectorizer
tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)


# In[7]:


from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브 베이즈 모델
mod = MultinomialNB()
model = mod.fit(tfidfv, train_data_intent) # 출력 결과에서 alpha=1.0은 라플라스 스무딩이 적용되었음을 의미


# In[9]:


from sklearn.metrics import accuracy_score # 정확도 계산을 위한 함수
X_test = data['tokens'][n_of_test : ]
y_test = data['intent'][n_of_test : ]

X_test_dtm = dtmvector.transform(X_test) # 테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(X_test_dtm) # DTM을 TF-IDF 행렬로 변환


# In[11]:


predicted = mod.predict(tfidfv_test) # 테스트 데이터에 대한 예측
print("정확도: ", accuracy_score(y_test, predicted)) # 테스트 데이터 예측값과 실제값 비교


# In[12]:


def predict(sentence):
    test_new = [sentence]
    test_dtm_new = dtmvector.transform(test_new)
    tfidfv_test_new = tfidf_transformer.transform(test_dtm_new)
    
    predicted_new = mod.predict(tfidfv_test_new)
    
    return predicted_new


# In[18]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_true = y_test
y_pred = mod.predict(tfidfv_test)

print(classification_report(y_true, y_pred, target_names = ['Flight', 'NoFlight']))


# In[ ]:


"""
import pickle

filename = 'MNB_model_1.sav'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(dtmvector, open('dtmvector_1', 'wb'))
pickle.dump(tfidf_transformer, open('tfidf_transformer_1', 'wb'))
"""


# In[ ]:


print(predict('new york arriving san farncisco'))
print(predict('on delta'))
print(predict('fare'))
print(predict('what type of aircraft'))
print(predict('all airport'))

print(predict('from new york to san francisco one way fare'))
print(predict('what ground transportation is available in san francisco'))


# In[ ]:




