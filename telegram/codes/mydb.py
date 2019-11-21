import pymysql

def getdb():
    mydb = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    db='flight',
    charset='utf8',
    autocommit=True)

    return mydb
