#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://api.telegram.org/bot682502305:AAHgR74_gRqpboE9VYFVsAOtAjZ5Tk1qtaw/deleteWebhook


# In[2]:


import telepot
import pickle
import sys
import time
import datetime
import re


# In[3]:


import import_ipynb
# github상에 올라와있는 _get_intent_MNB와 _get_intent_LSTM 중 하나를 선택하여 이름을 _get_intent로 수정하여 실행하시면 됩니다.
from _get_intent import get_intent
from spelling_corrector import _correction
from _get_tag import get_element, todf, ellipsis
from mydb import getdb
from _get_view import getroute
from _todatefomat import date as _date


# In[4]:


token = "682502305:AAHgR74_gRqpboE9VYFVsAOtAjZ5Tk1qtaw"
my_id = "729092704"
bot = telepot.Bot(token)
status = True
InfoMsg = "WELCOME"
Answer = "SEE YOU AGAIN"
state = 0
flag = 9


# In[5]:


intent = None
tagged = None
def handle(msg):
    global flag
    global intent
    global tagged
    global eFlag # ellipsis 여부
    global user_input
    
    content_type, chat_type, chat_id = telepot.glance(msg)

    if msg['text'].lower() in ['hi', 'hello', 'airscope', 'hey']:
        flag = 9
        eFlag=0
        
    if flag == 9:
        if content_type == 'text':
            if msg['text'].lower() in ['hi', 'hello', 'airscope', 'hey']:
                bot.sendMessage(chat_id, InfoMsg)
                eFlag=0
            elif msg['text'].lower() in ['bye', 'see you', 'thank you', 'thanks']:
                bot.sendMessage(chat_id, Answer)
                eFlag = 0
            else:
                if eFlag == 0:
                    user_input = _correction(msg['text'])
                    #bot.sendMessage(chat_id, user_input)
                    bot.sendMessage(chat_id,"Looking for results...")
                    intent = get_intent(user_input)
                    if intent==False:
                        bot.sendMessage(chat_id,"Sorry, I can't answer that.")
                    intent = ''.join(intent)
                    #bot.sendMessage(chat_id, intent)
                    tagged = get_element(user_input, intent)
                    flag = 0
                    eFlag = 1
                else : #ellipsis 경우
                    second_input = msg['text']
                    bot.sendMessage(chat_id,"Looking for results...")
                    new_sent, tagged2 = ellipsis(tagged, user_input, second_input)
                    intent2 = get_intent(new_sent)
                    flag=7


    if flag == 1:
        tagged.element[1] = msg['text']
        flag=0
    elif flag == 2:
        tagged.element[3] = msg['text']
        flag=0
    elif flag == 3:
        date = _date(todf(msg['text']))
        tagged.element[4]=date
        flag=0
    elif flag == 4:
        tagged.element[0]=msg['text']
        flag=0
    elif flag == 5:
        reply = msg['text'].lower()
        if "cheapest" in reply:
            tagged.element[7]=1
            flag=0
        else:
            if reply.isdigit:
                tagged.element[6]='\''+re.findall('\d+', reply)[0]+'\''
                flag=0
            else:
                bot.sendMessage(chat_id,'{}'.format("Sorry, I didn't get that. Below how much?"))
                reply=msg['text']
                tagged.element[6]='\''+re.findall('\d+', reply)[0]+'\''
                flag=0
    else:
        flag=flag
    
    if flag == 0 :
        # if null값 있으면 flag=1, 모두 있으면 flag=2
        if tagged[tagged['tag'].isin(['fromloc'])].element.tolist()[0]=='':
            bot.sendMessage(chat_id,'{}'.format("Where would you like to depart?"))
            flag = 1
        elif tagged[tagged['tag'].isin(['toloc'])].element.tolist()[0]=='':
            bot.sendMessage(chat_id,'{}'.format("Where would you want to arrive?"))
            flag = 2
        elif tagged[tagged['tag'].isin(['dpttime'])].element.tolist()[0]=='' and tagged[tagged['tag'].isin(['arrtime'])].element.tolist()[0]=='':
            bot.sendMessage(chat_id,'{}'.format("When do you want to depart?"))
            flag = 3  
        elif 'AskFlightWithAirline' in intent and tagged[tagged['tag'].isin(['airline'])].element.tolist()[0]=='':
                bot.sendMessage(chat_id, '{}'.format('Which airline would you like'))
                flag = 4
        elif 'AskFlightWithCost' in intent and tagged[tagged['tag'].isin(['cheapest'])].element.tolist()[0]==0 and tagged[tagged['tag'].isin(['cost'])].element.tolist()[0]=='':
                bot.sendMessage(chat_id, '{}'.format("Do you want the cheapest flight or a ticket below a certain price?"))
                flag = 5
        else: 
            flag=6

    if flag==6:
        mydb = getdb()
        getroute(intent, tagged, mydb, bot, chat_id)
        flag = 9
        
    if flag==7:
        mydb = getdb()
        getroute(intent2, tagged2, mydb, bot, chat_id)
        flag = 9
        
bot.message_loop(handle)

while status == True:
    time.sleep(10)


# In[ ]:




