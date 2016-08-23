__author__ = "Arkenstone"

import os
import sys
from connect_db import extractDataFromDB
from customer_behavior_functions import calculate_time_interval
from os.path import isfile, isdir, exists
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime as dt

class trainingSetSelection():
    def __init__(self,
                 localhost="yourDBIP",
                 username="yourUserName",
                 password="yourPassWord",
                 dbname="yourDB",
                 trans_tbname="yourTransactionTable",
                 enter_tbname="yourEnterpriseTable",
                 training_set_times_range=(3, np.inf),
                 training_set_length=3,
                 init_date=dt.datetime.now()-dt.timedelta(365*2),
                 cus_threshold=100
                 ):
        """
        enterprise table contains all enterprise id info, following such format:
        enterprise_id | city | address  | ...
        1             | SZ   | xxx road | ...
        ________________________________________________________________________
        transaction table contains all transaction info, following such format:
        transaction_id | customer_id | enterprise_id | price | create_time | ...
        1              | 1           | 1             | 10    | 2016-01-01  | ...
        ________________________________________________________________________
        :param training_set_times_range: customers analyzed whose purchase times is with in this range will be retrieved
        :param training_set_length=3: number of customers in each training set input
        :param init_date: only transactions after this date will be used. default: start from 1 year ago
        :param cus_threshold: only enterprise whose number of customers that could be analyzed reach the threshold will be analyzed
        """
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.trans_tbname = trans_tbname
        self.enter_tbname = enter_tbname
        self.training_set_times_range = training_set_times_range
        self.training_set_length = training_set_length
        self.init_date = init_date
        self.threshold = cus_threshold

    def statistic_db_enterprise_transaction_distribution(self, outfile="filtered_enterprise_id_list.csv"):
        """
        :param outfile="filtered_enterprise_id_list.csv": output filtered enterprise to this file
        :return: output enterprise id whose frequent customer count (default 4) is larger than threshold (default 100) to a file
        """
        print "Scanning all enterprises transaction data to filter enterprises whose number of frequent customer reach the minimum threshold ..."
        print "%s: Start statistic..." %str(dt.datetime.now())
        currentDB = extractDataFromDB()
        currentDB.localhost = self.localhost
        currentDB.username = self.username
        currentDB.password = self.password
        currentDB.dbname = self.dbname
        # get enterprise_id list in the enterprise db
        print "Retrieving enterprise id list from enterprise table ..."
        currentDB.tbname = self.enter_tbname
        db_cursor = currentDB.connect_db()
        enterprise_df = currentDB.get_data_from_db(db_cursor=db_cursor,
                                                   selected=["enterprise_id"],
                                                   filter=["create_time > '" + str(self.init_date) + "'"])
        # connect to transaction db
        currentDB.tbname = self.trans_tbname
        db_cursor = currentDB.connect_db()
        # filter enterprises
        dict_fil_enter = {}
        for enterprise in enterprise_df.enterprise_id:
            print "Analyzing current enterprise: %s" %str(enterprise)
            df = currentDB.get_data_from_db(db_cursor=db_cursor,
                                            selected=["customer_id", "create_time"],
                                            filter=["create_time > '" + str(self.init_date) + "'", "enterprise_id = " + str(enterprise)])
            # next loop if df is empty:
            if df.empty:
                continue
            # remove duplicates of a customer in same day
            df.create_time = df.create_time.apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
            df = df.drop_duplicates(['customer_id', 'create_time'])
            cus_count = ((df.customer_id.value_counts() >= self.training_set_times_range[0]+1) & (df.customer_id.value_counts() <= self.training_set_times_range[1]+1)).sum()
            if cus_count >= self.threshold:
                dict_fil_enter[enterprise] = cus_count
                print "Satisfied!"
        print "Analyzing enterprise done!"
        # convert dict to df
        currentDB.disconnect_db(db_cursor)
        df_fil_enter = pd.DataFrame.from_dict(dict_fil_enter, orient="index")
        df_fil_enter.columns = ['customer_count']
        df_fil_enter['enterprise'] = df_fil_enter.index
        # sort df according to the enterprise ids
        df_fil_enter = df_fil_enter.sort_values(['enterprise'], ascending=True)
        print "Write filtered enterprises to file: %s" %str(outfile)
        df_fil_enter.to_csv(outfile)
        print "%s: End statistic!" %str(dt.datetime.now())

    def check_transaction_data(self, transaction_df, init_date):
        """
        :param transaction_df: df with transaction data, must have customer_id and create_time
        :param total_transaction_times: minimum transactions times that a customer should have: default: 5
                                        (4 intervals - 3 for training set and one for test)
        :param init_date: only transactions after this time will be used. default: 1 year ago to current date
        :return:transaction_df after init date
        """
        # check the input init_date format
        if isinstance(init_date, dt.datetime):
            pass
        elif isinstance(init_date, str):
            init_date = parse(init_date)
        else:
            raise TypeError("init_date must be datetime type or str type!")

        # get transactions after init time
        earliest = np.min(transaction_df.create_time)
        latest = np.max(transaction_df.create_time)
        if init_date > latest:
            raise ValueError("Init_date is too late! It should not surpass the last transaction date!")
        else:
            transaction_df = transaction_df.ix[transaction_df.create_time >= init_date, ]

        # filter customer whose transaction times is larger than requirement
        # get customer_ids match requirement
        transaction_df.create_time = transaction_df.create_time.apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
        transaction_df = transaction_df.drop_duplicates(['customer_id', 'create_time'])
        cus_trans_count = transaction_df.customer_id.value_counts().index[transaction_df.customer_id.value_counts() >= self.training_set_times_range[0]+1].tolist()
        transaction_df = transaction_df.ix[transaction_df.customer_id.isin(cus_trans_count), :]
        return transaction_df

    def trainingSetGeneration(self, outdir=".", override=False, **kwargs):
        """
        :param outdir=".": output directory for generated training set
        :param overide=False: override existed enterprise id list file and training set files
        :param kwargs: enterprise_id_list="filtered_enterprise_id_list.csv": contains enterprise ids whose dataset meet the minimum requirement.
                            If not provided, the statistic_db_enterprise_transaction_distribution will be performed
                            and the default file is a csv file "called filtered_enterprise_id_list.csv" in current diretory;
                       selected = ["customer_id", "enterprise_id", "price", "create_time"];
                       outfile="filtered_enterprise_id_list.csv"
        :return: output training set files corresponding to each filtered enterprise
        Note: 1. Total transaction period should be larger than traning_set_times + 1 (test_set_times)
              2. if init date is not current date, it should follow time format: yyyy-mm-dd
        """
        def create_interval_dataset(dataset, look_back):
            """
            :param dataset: input array of time intervals
            :param look_back: each training set feature length
            :return: convert an array of values into a dataset matrix.
            """
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back):
                dataX.append(dataset[i:i+look_back])
                dataY.append(dataset[i+look_back])
            return np.asarray(dataX), np.asarray(dataY)

        print "Get training dataset of each enterprise..."
        print "%s: Start generation..." %str(dt.datetime.now())
        enterprise_id_list_file = kwargs.get('enterprise_id_list', "filtered_enterprise_id_list.csv")
        selected = kwargs.get('selected', ["customer_id", "enterprise_id", "create_time", 'price'])

        # create output dir if not exists
        if not exists(outdir):
            os.makedirs(outdir)
        # get enterprise id list is provided if it is not provided
        enter_file = outdir+'/'+enterprise_id_list_file
        if not isfile(enter_file) or override:
            self.statistic_db_enterprise_transaction_distribution(outfile=enter_file)
        print "Get candidate enterprise from file: %s" %str(enter_file)
        # get the enterprise id list
        df_enter = pd.read_csv(enter_file)
        enterprise = df_enter.enterprise
        # connect to the transaction db
        currentDB = extractDataFromDB()
        currentDB.localhost = self.localhost
        currentDB.username = self.username
        currentDB.password = self.password
        currentDB.dbname = self.dbname
        currentDB.tbname = self.trans_tbname
        db_cursor = currentDB.connect_db()
        for currentEnterprise in enterprise:
            outfile = outdir + "/" + str(currentEnterprise) + ".csv"
            # override the existing file or not
            interval_file = outdir + "/" + str(currentEnterprise) + ".intervals.csv"
            if exists(interval_file) and not override:
              continue
            print "Retrieving transaction data of %s from transaction table" %str(currentEnterprise)
            trans_df = currentDB.get_data_from_db(db_cursor=db_cursor,
                                                  selected=selected,
                                                  filter=["create_time > '" + str(self.init_date) + "'", "enterprise_id = " + str(currentEnterprise)])
            # df with interval
            df_interval = calculate_time_interval(trans_df)
            # remove lines with time interval is 0
            df_interval = df_interval.ix[df_interval.time_interval > 0, :]
            # output intervals data to file for later distribution assessment and data merging
            interval_output = df_interval.time_interval
            interval_output.to_csv(interval_file)
            if exists(outfile) and not override:
               continue
            # filter customers whose transaction intervals overpass the minimum requirement: training set count + 1
            # cus_trans_count = df_interval.customer_id.value_counts().index[df_interval.customer_id.value_counts() >= self.training_set_times + 1].tolist()
            # df_interval = df_interval.ix[df_interval.customer_id.isin(cus_trans_count), :]
            print "Filtering customers whose purchase times meet the minimum threshold: %s" %str(self.threshold)
            df_interval = self.check_transaction_data(df_interval, init_date=self.init_date)
            # get all unique customer_ids
            all_cus_ids = df_interval.customer_id.unique()
            df_cur_enter = pd.DataFrame()
            print "Formating the dataset..."
            for currentCustomer in all_cus_ids:
                dataset = df_interval.time_interval[df_interval.customer_id == currentCustomer]
                dataset = np.asarray(dataset)
                dataX, dataY = create_interval_dataset(dataset, look_back=self.training_set_length)
                X_cols = []
                for x in range(1, 1+self.training_set_length):
                    X_cols.append('X' + str(x))
                dfX = pd.DataFrame(dataX, columns=X_cols)
                dfY = pd.DataFrame(dataY, columns=['Y'])
                dfY['customer_id'] = currentCustomer
                dfY['enterprise_id'] = currentEnterprise
                df_cur_cus = pd.concat((dfX, dfY), axis=1)
                df_cur_enter = pd.concat((df_cur_enter, df_cur_cus), axis=0)
            # output training dataset of current enterprise to output directory
            print "Output formated training dataset to file: %s" %str(outfile)
            # reindex the output file
            df_cur_enter.index = range(len(df_cur_enter.index))
            df_cur_enter.to_csv(outfile)
        print "%s: End generation!" %str(dt.datetime.now())

if __name__ == "__main__":
    outdir = "path-of-your-output-directory"
    if not exists(outdir):
        os.makedirs(outdir)
    # redirect stdout to log file
    old_stdout = sys.stdout
    logfile = open(outdir + "/message.log.txt", "w")
    print "Log message could be found in file: %s" %str(logfile)
    sys.stdout = logfile
    obj_trainingSet.training_set_times_range = (10, np.inf)
    obj_trainingSet.training_set_length = 5
    obj_trainingSet.init_date = dt.datetime.now() - dt.timedelta(365*2)
    obj_trainingSet.threshold = 10
    obj_trainingSet.trainingSetGeneration(outdir)
    # return to normal stdout
    sys.stdout = old_stdout
    logfile.close()
