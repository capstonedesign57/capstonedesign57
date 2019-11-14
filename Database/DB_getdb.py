#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pymysql


# In[5]:


def getdb():
    mydb = pymysql.connect(
    host = 'localhost',
    port = 3306,
    user = 'root',
    passwd = '', # Insert your password
    db = 'airscope', # Insert your Database name
    charset = 'utf8',
    autocommit = True)
    
    return mydb

