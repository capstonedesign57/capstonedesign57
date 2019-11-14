#!/usr/bin/env python
# coding: utf-8

# # 양방향 LSTM과 CRF(Bidirectional LSTM + CRF)

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv('BIO_TAGGER_TRAINING_DATASET.csv')


# In[2]:


data = data.fillna(method = "ffill") # Null값 
data.head(5)


# In[3]:


# 단어의 소문자화
data['Word'] = data['Word'].str.lower()
data.head(5)


# In[4]:


# 중복을 허용하지 않고, 단어들을 모아 단어 집합을 만듦
# 단어 dictionary 만들기 
vocab = (list(set(data["Word"].values)))
print(vocab)


# In[5]:


# 중복을 허용하지 않고, 태그들을 모아 태그 집합을 만듦
# 태그 dictionary 만들기
tags = list(set(data["Tag"].values))
print(len(tags))


# In[6]:


data.head(5)


# In[7]:


# 하나의 문장에 등장한 단어와 개체명 태깅 정보끼리 pair로 묶음
# example : (what, O)

#temp의 "Word" 컬럼의 값들을 리스트로 ( temp["Word"].values.tolist() )
#tepm의 "Tag" 컬럼의 값들을 리스트로 ( temp["Tag"].values.tolist() ) 만들어 zip하고 
#각 리스트의 값들을 순서대로 하나씩 가져와 각각 w, t로 하여 pair (w,t)를 만든다 
func = lambda temp: [(w, t) for w, t in zip(temp["Word"].values.tolist(), temp["Tag"].values.tolist())]
#data의 'Sentence #'컬럼의 값별로 그룹화하여 func 적용한다
All_data=[t for t in data.groupby("Sentence #").apply(func)]


# In[8]:


All_data[0]


# In[9]:


# 단어와 개체명 태깅 정보를 분리하는 작업

import numpy as np
sentences, ner_tags = [], []
for tagged_sentence in All_data:
    sentence, ner_info = zip(*tagged_sentence) # 각 샘플에서 단어는 sentence에, 개체명 태깅 정보는 ner_infodp
    sentences.append(np.array(sentence)) # 각 샘플에서 단어 정보만 저장
    ner_tags.append(np.array(ner_info)) # 각 샘플에서 개체명 태깅 정보만 저장


# In[10]:


# example
sentences[0]


# In[11]:


# example
ner_tags[0]


# In[12]:


# 단어 집합과 개체명 태깅 정보의 집합 만들기
# 단어 집합의 경우 Counter()를 이용해 단어의 등장 빈도수 계산

from collections import Counter
vocab = Counter()
tag_set = set()
for sentence in sentences: # sentences에서 샘플을 1개씩 꺼내온다.
    for word in sentence: # sentence에서 단어를 1개씩 꺼내온다.
        vocab[word.lower()] = vocab[word.lower()] + 1 # 각 단어의 빈도수 계산
        
for tags_list in ner_tags: # ner_tage에서 샘플을 1개씩 꺼내온다.
    for tag in tags_list: # tags_list에서 개체명 정보를 1개씩 꺼내온다.
        tag_set.add(tag) # 각 개체명 정보에 대해서 중복을 허용하지 않고 집합을 만든다.


# In[13]:


vocab


# In[14]:


len(tag_set)


# In[15]:


# 단어 집합을 등장 빈도수를 기준으로 정렬
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)


# In[16]:


vocab_sorted


# In[17]:


word_to_index = {'PAD' : 0, 'OOV' :1}
i = 1
# 인덱스 0은 각각 입력값들의 길이를 맞추기 위한 PAD(padding을 의미)라는 단어에 사용된다.
# 인덱스 1은 모르는(단어 dictionary에 없는) 단어를 의미하는 OOV라는 단어에 사용된다.
for (word, frequency) in vocab_sorted :
        i = i + 1
        word_to_index[word] = i


# In[18]:


#빈도수로 나열된 단어별로 인덱스 부여
word_to_index


# In[19]:


# 태깅 정보에도 인덱스를 부여
tag_to_index = {'PAD' : 0}
i = 0
for tag in tag_set:
    i = i + 1
    tag_to_index[tag] = i


# In[20]:


tag_to_index


# In[21]:


# 단어 정수 인코딩 진행
data_X = []

for s in sentences: # 전체 데이터에서 하나의 데이터. 즉, 하나의 문장씩 불러옵니다.
    temp_X = []
    for w in s: # 각 문장에서 각 단어를 불러옵니다.
        temp_X.append(word_to_index.get(w,1)) # 각 단어를 매핑되는 인덱스로 변환합니다.
    data_X.append(temp_X)


# In[22]:


data_X[0]


# In[23]:


# 개체명 태깅 정보 정수 인코딩 진행
data_y = []
for s in ner_tags:
    temp_y = []
    for w in s:
            temp_y.append(tag_to_index.get(w))
    data_y.append(temp_y)


# In[24]:


data_y[0]


# In[25]:


# 패딩
max_len = 45
from keras.preprocessing.sequence import pad_sequences
pad_X = pad_sequences(data_X, padding = 'post', maxlen = max_len)
# data_X의 모든 샘플의 길이를 패딩할 때, 뒤의 공간을 숫자 0으로 채움
pad_y = pad_sequences(data_y, padding = 'post', value = tag_to_index['PAD'], maxlen = max_len)
# data_y의 모든 샘플의 길이를 패딩할 때, 'PAD'에 해당하는 인덱스로 채움
# 결과적으로 'PAD'의 인덱스 값인 0으로 패딩됨


# In[26]:


pad_X[0]


# In[27]:


pad_y[0]


# In[28]:


# 훈련 데이터와 테스트 데이터를 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size = .1, random_state = 777)

# 단어 집합과 태깅 정보 집합의 크기를 변수에 저장
# 모델 생성에 사용할 변수
n_words = len(word_to_index)
n_labels = len(tag_to_index)


# In[29]:


# 모델에 양방향 LSTM을 사용, 모델의 출력층에 CRF층을 배치
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

model = Sequential()
model.add(Embedding(input_dim = n_words, output_dim = 20, input_length = max_len, mask_zero = True))
model.add(Bidirectional(LSTM(units = 50, return_sequences = True, recurrent_dropout = 0.1)))
model.add(TimeDistributed(Dense(50, activation = "relu")))
crf = CRF(n_labels)
model.add(crf)


# In[30]:


from keras.utils import np_utils
y_train2 = np_utils.to_categorical(y_train) # one-hot 인코딩


# In[31]:


pd.set_option('max_colwidth', 800)


# In[32]:


y_train2[0]


# In[33]:


print(y_train[0])


# In[34]:


model.compile(optimizer = "rmsprop", loss = crf.loss_function, metrics = [crf.accuracy])
history = model.fit(X_train, y_train2, batch_size = 32, epochs = 20, validation_split = 0.1, verbose = 1)


# In[35]:


y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test2)[1]))


# In[36]:


index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

index_to_tag = {}
for key, value in tag_to_index.items():
    index_to_tag[value] = key


# In[37]:


# 실제 모델에 대해 f1 score 구하기
def sequences_to_tag(sequences): # 예측값을 index_to_tag를 사용하여 태깅 정보로 변경하는 함수.
    result = []
    for sequence in sequences: # 전체 시퀀스로부터 시퀀스를 하나씩 꺼낸다.
        temp = []
        for pred in sequence: # 시퀀스로부터 예측값을 하나씩 꺼낸다.
            pred_index = np.argmax(pred) # 예를 들어 [0, 0, 1, 0 ,0]라면 1의 인덱스인 2를 리턴한다.
            temp.append(index_to_tag[pred_index].replace("PAD", "O")) # 'PAD'는 'O'로 변경
        result.append(temp)
    return result


# # Test

# In[38]:


from nltk import word_tokenize


# In[39]:


test_sentence = "Show me the flight which arriving new york in this friday"
test_sentence = test_sentence.lower()
test_sentence = word_tokenize(test_sentence)


# In[40]:


new_X = []
for w in test_sentence:
    try:
        new_X.append(word_to_index.get(w,1))
    except KeyError:
        new_X.append(word_to_index['OOV'])
      # 모델이 모르는 단어에 대해서는 'OOV'의 인덱스인 1로 인코딩


# In[41]:


pad_new = pad_sequences([new_X], padding="post", value=0, maxlen=max_len)


# In[42]:


p = model.predict(np.array([pad_new[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(test_sentence, p[0]):
    print("{:15}: {:5}".format(w, index_to_tag[pred]))


# In[43]:


#모델 저장
model.save("_BIO_TAGGER_FINAL.h5")


# In[44]:


#dictionary 저장
import pickle
with open('_word_to_index_FINAL.pickle','wb') as f:
    pickle.dump(word_to_index, f)


# In[45]:


with open('_index_to_tag_FINAL.pickle','wb') as f:
    pickle.dump(index_to_tag, f)


# In[46]:


#with open('X_test_1_TIME_2.pickle','wb') as f:
 #   pickle.dump(X_test, f)
#with open('y_test_1_TIME_2.pickle','wb') as f:
 #   pickle.dump(y_test, f)


# In[ ]:




