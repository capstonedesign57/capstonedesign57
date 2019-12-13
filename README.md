# capstonedesign57
## jupyter version link => https://github.com/capstonedesign57/AirScope_jupyterVer.git
AirScope_jupyterVer
====================
AirScope는 자연어로 주어진 사용자의 질의에서 의도와 개체명을 정확히 추출하여 조건을 만족하는 항공권을 반환하는 챗봇이다.
---
AirScope is a chatbot that provides Air Travel Information based on Natural Language Processing. It predicts the user's intent and extracts the named entities and returns the flight information that meets the requirements.
---
# 1. Collecting Data
We collected data from the kaggle site (https://www.kaggle.com/siddhadev/atis-dataset-clean) and did some manual work to change the intent into intents we use.
# 2. Intent Classifier
## 1) Flight/NoFlight
This is the first model of the two intent classifiers we built. When the user types in a sentence, this model returns either flight or noflight. Both the MNB model and LSTM model uses the same data for training which is train_flight_noflight.csv file.
### a) Multinomial Naive Bayes
We used CountVectorizer and TfIdfTransformer provided by sklearn library to embed english text.
### b) Long Short-Term Memory
We used pretrained FastText model which we downloaded from the FastText page (https://fasttext.cc/docs/en/crawl-vectors.html)
## 2) Detail Intent ( Only if the User's intent is Flight )
This is the second model of the two intent classifiers we built. The Model predicts the detailed intent from the four intents we used to train the model which are flight(depart, arriving location, time), flight + cost, flight + airline, flight + cost + airline.
### a) Long Short-Term Memory
We used pretrained FastText model which we downloaded from the FastText page (https://fasttext.cc/docs/en/crawl-vectors.html). 
# 3. Named Entity Recognition
## 1) Bidirectional Long Short-Term Memory + Conditional Random Field
We refered to the wikidocs page ( https://wikidocs.net/34156 ) for implementing the main training code and used atis data collected from the kaggle site. 
# 4. Bot code using Telegram API
## 1) Telepot Library
We implemented our chatbot on telegram application and used telepot(telegram api) which is a python library.
## 2) spell corrector
We refered to the Norvig method code from a site(http://theyearlyprophet.com/spell-correct.html).
## 3) get_intent
We used two models we built earlier. If the first model returns noflight, the chatbot notifies the user that the chatbot can't handle the question. If the first model returns flight, the second model predicts the detailed intent.
## 4) get_tag
We use the BiLSTM-CRF model we built earlier. First, the model tags each tokens and returns it as a dataFrame. Then the get_element method extracts the key elements that are needed for building SQL query. There is also ellipsis function which handles additional input.
## 5) get_view
Using the SQL query built based on the intent from get_intent file and the elements from get_tag file, the chatbot gets the flight information from the database which we manually built based on search results from Google and the Skyscanner page(https://www.skyscanner.co.kr/?ksh_id=_k_CjwKCAiAxMLvBRBNEiwAKhr-nPHZ1M-7glQCwXn7nJygeVHS69Ulcc0CI4gxCd73vpMRpMxznEDbjxoCAsMQAvD_BwE_k_&associateID=SEM_GGT_00065_00021&utm_source=google&utm_medium=cpc&utm_campaign=KR-Travel-Search-Brand-Pure-Exact&utm_term=skyscanner&kpid=google_6464657269_76880054785_379362472269_kwd-400074527_c_&gclid=CjwKCAiAxMLvBRBNEiwAKhr-nPHZ1M-7glQCwXn7nJygeVHS69Ulcc0CI4gxCd73vpMRpMxznEDbjxoCAsMQAvD_BwE).
