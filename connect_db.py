__author__ = "Arkenstone"

import MySQLdb as msdb
import datetime as dt
import pandas as pd
import numpy as np
import random

# class for connect to database
class extractDataFromDB:
    def __init__(self, localhost="yourIP", username="yourUserName", password="yourPassword", dbname="yourDB", tbname="yourTable"):
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname

    def connect_db(self):
        # connect to the database
        db = msdb.connect(host=self.localhost, user=self.username, passwd=self.password, db=self.dbname)
        db_cursor = db.cursor()
        # return a db cursor
        return db_cursor

    def disconnect_db(self, cursor):
        cursor.close()

    def get_data_from_db(self, db_cursor, selected, filter=None):
        # input: columns title need to be retrieved;
        # output: columns retrieved
        ### NOTE: filter should be list format: ["create_time < '2016-06-02'", "enterprise_id = 256"] ####
        # and selected should be list like ['customer_id', 'create_time']
        # choose table
        tbname = self.tbname
        # choose items
        outID = selected
        selected = ', '.join(selected)
        # filter conditions if exist
        if filter:
            cond = ' and '.join(filter)
            # sql filtering command
            sql = "SELECT " + selected + " FROM " + tbname + " WHERE " + cond
        else:
            sql = "SELECT " + selected + " FROM " + tbname
        # initial a dictionary for holding the data
        my_data = {}
        try:
            # fetch all data selected
            db_cursor.execute(sql)
            results = db_cursor.fetchall()
            # exit the function if the return results tuple is empty
            if not results:
                return pd.DataFrame()
            count = 0
            for row in results:
                my_data[count] = row
                count += 1
        except:
            print "Error: cannot fetch data from %s" %(tbname)
        # convert the data in dictionary ro data frame
        df = pd.DataFrame.from_dict(my_data, orient='index')
        df.columns = outID
        return df
