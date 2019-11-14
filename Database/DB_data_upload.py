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


# In[4]:


import pandas as pd
import csv


# In[6]:


# cursor.execute("DELETE FROM table명")
db.commit()


# # airlines

# In[7]:


airlines = pd.read_csv('airlines.csv')


# In[8]:


airlines = airlines[["airline_id", "name", "iata", "icao", "country"]]


# In[9]:


airlines.head()


# In[11]:


for i in range(len(airlines)):
    cursor.execute(
        "INSERT INTO \
        airlines(airline_id, name, iata, icao, country) \
        VALUES (%s, %s, %s, %s, %s)", \
        (int(airlines.iloc[i][0]), airlines.iloc[i][1], airlines.iloc[i][2], \
        airlines.iloc[i][3], airlines.iloc[i][4])
    )
db.commit()


# # airports

# In[12]:


airports = pd.read_csv('airports.csv')
airports = airports[["airport_id", "name", "city", "country", "iata"]]
airports = airports.dropna(axis=0)


# In[13]:


for i in range(len(airports)):
    cursor.execute(
            "INSERT INTO \
            airports(airport_id, name, city, country, iata) \
            VALUES (%s,%s,%s,%s,%s)", \
            (int(airports.iloc[i][0]), airports.iloc[i][1], airports.iloc[i][2], \
            airports.iloc[i][3], airports.iloc[i][4])
    )
db.commit()


# # routes

# In[14]:


routes = pd.read_csv('routes.csv')


# In[15]:


for i in range(len(routes)):
    cursor.execute(
        "INSERT INTO \
        routes(route_id, airline, airline_id, src_airport, src_airport_id, \
        dst_airport, dst_airport_id, stops, dpt_time, est_time) \
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", \
        (int(routes.iloc[i][0]), routes.iloc[i][1], int(routes.iloc[i][2]), \
         routes.iloc[i][3], int(routes.iloc[i][4]), routes.iloc[i][5], \
         int(routes.iloc[i][6]), int(routes.iloc[i][7]), routes.iloc[i][8], \
         routes.iloc[i][9])
    )
db.commit()


# # stops

# In[16]:


stops = pd.read_csv('stops.csv')


# In[17]:


for i in range(len(stops)):
    cursor.execute(
        "INSERT INTO \
        stops(route_id, stop, stop_id, stop_order) \
        VALUES (%s, %s, %s, %s)", \
        (int(stops.iloc[i][0]), stops.iloc[i][1], int(stops.iloc[i][2]), \
        int(stops.iloc[i][3]))
    )
db.commit()


# # cost

# In[18]:


cost = pd.read_csv('cost.csv')


# In[19]:


for i in range(len(cost)):
    cursor.execute(
        "INSERT INTO \
        cost(route_id, airline_id, cost) \
        VALUES (%s, %s, %s)", \
        (int(cost.iloc[i][0]), int(cost.iloc[i][1]), int(cost.iloc[i][2]))
    )
db.commit()

