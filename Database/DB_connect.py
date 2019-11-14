#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib
import pymysql
from urllib.request import urlopen
from bs4 import BeautifulSoup


# In[2]:


db = pymysql.connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '', # Insert your password
    db = 'airscope', # Insert your Database name
    charset = 'utf8',
    autocommit = True)


# In[3]:


cursor = db.cursor()

cursor.execute("SELECT VERSION()")

data = cursor.fetchone()

print("Database version : %s " % data)

db.close()

