__author__ = "Arkenstone"

import os
import sys
from preprocessing import read_df_from_mysql_db, list2file, file2list
from dateutil.parser import parse
import pandas as pd
import numpy as np
import datetime as dt

class TrainingSetSelection():
    def __init__(self,
                 localhost="112.74.30.59",
                 username="fanzong",
                 password="maxfun",
                 dbname="maxfun_tp",
                 trans_tbname="transaction",
                 enter_tbname="enterprise",
                 enter_field="enterprise_id",
                 enter_list=None,
                 min_purchase_count=4,
                 train_input_length=3,
                 init_date=dt.datetime.now()-dt.timedelta(180),
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
        :param enter_field: enterprise_id header in mysql table
        :param enter_list: specify enterprise_id list to retrieve
        :param min_purchase_count: customers above this minimum purchase count will be analyzed
        :param cus_threshold: enterprise above minimum regular customers will be analyzed
        :param train_input_length=3: number of customers in each training set input
        :param init_date: only transactions after this date will be used. default: start from 1 year ago
        """
        self.localhost = localhost
        self.username = username
        self.password = password
        self.dbname = dbname
        self.trans_tbname = trans_tbname
        self.enter_tbname = enter_tbname
        self.enter_field = enter_field
        self.enter_list = enter_list
        self.min_purchase_count = min_purchase_count
        self.train_input_length = train_input_length
        # assert train_input length must be smaller than min_purchase _count
        try:
            assert min_purchase_count > train_input_length
        except:
            raise ValueError("min_puchase_count should be larger than train_input_length! Reset your input.")
        self.init_date = init_date
        self.customer_threshold = cus_threshold

    def _select_enterprises(self):
        print "Scanning all enterprises transaction data to filter enterprises whose number of frequent customer reach the minimum threshold ..."
        # get enterprise_id list in the enterprise db
        print "Retrieving enterprise id list from enterprise table ..."
        enterprises_id_df = next(read_df_from_mysql_db(localhost=self.localhost, username=self.username, password=self.password, dbname=self.dbname, tbname=self.enter_tbname, enter_field=self.enter_field, enterprise_id=self.enter_list, fields="enterprise_id"))
        enterprises_trans_df = next(read_df_from_mysql_db(localhost=self.localhost, username=self.username, password=self.password, dbname=self.dbname,
                                              tbname=self.trans_tbname, enter_field=self.enter_field, enterprise_id=self.enter_list, fields=["customer_id", "enterprise_id", "create_time"],
                                              time_field="create_time", start_time=self.init_date.strftime("%Y-%m-%d")))

        # filter enterprises
        filter_enters = []
        for enterprise_id in enterprises_id_df.enterprise_id:
            print "Analyzing current enterprise: {}".format(enterprise_id)
            enter_df = enterprises_trans_df[enterprises_trans_df['enterprise_id'] == enterprise_id]
            # next loop if df is empty:
            if len(enter_df.index) == 0:
                continue
            # remove duplicates of a customer in same day
            enter_df.create_time = enter_df.create_time.apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
            try:
                assert 'customer_id' in enter_df.columns and 'create_time' in enter_df.columns
            except:
                raise ValueError("Input df must have customer_id header and crete_time header!")
            enter_df = enter_df.drop_duplicates(['customer_id', 'create_time'])
            cus_count = (enter_df.customer_id.value_counts() >= self.min_purchase_count).sum()
            if cus_count >= self.customer_threshold:
                filter_enters.append(enterprise_id)
                print "enterprise {} satisfied: {} customers.".format(enterprise_id, cus_count)
        print "Analyzing enterprise done!"
        return filter_enters


    def check_transaction_data(self, transaction_df, init_date):
        # transaction_df after init date
        # check the input init_date format
        if isinstance(init_date, dt.datetime):
            pass
        elif isinstance(init_date, str):
            init_date = parse(init_date)
        else:
            raise TypeError("init_date must be datetime type or str type!")
        # get transactions after init time
        latest_date = np.max(transaction_df.create_time)
        if init_date > latest_date:
            raise ValueError("Init_date is too late! It should not surpass the last transaction date!")
        else:
            transaction_df = transaction_df.ix[transaction_df.create_time >= init_date, ]
        # filter customer whose transaction times is larger than requirement
        # get customer_ids match requirement
        transaction_df.create_time = transaction_df.create_time.apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
        transaction_df = transaction_df.drop_duplicates(['customer_id', 'create_time'])
        # get index in df of customers meeting the count requirement
        good_cus_index = transaction_df.customer_id.value_counts().index[transaction_df.customer_id.value_counts() >= self.min_purchase_count].tolist()
        transaction_df = transaction_df.ix[transaction_df.customer_id.isin(good_cus_index), :]
        return transaction_df


    def _calculate_time_interval(self, df):
        # assert customer id and create time header in df
        try:
            assert 'customer_id' in df.columns
            assert 'create_time' in df.columns
        except:
            raise ValueError("customer_id and create_time header is not in your input df!")
        # sort df in ascending order by customer id firstly and create time secondly
        df_sorted = df.sort_values(['customer_id', 'create_time'], ascending=[True, True])
        cus = df_sorted['customer_id']
        # Only use day time
        time = df_sorted['create_time'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
        previous_cus = cus.iloc[0]
        previous_time = time.iloc[0]
        # initialize interval pd series
        time_interval = pd.DataFrame(np.zeros(len(cus)))
        time_interval.index = df_sorted.index
        # initial a dictionary for holding average freq
        for i in range(1, len(cus)):
            cur_cus = cus.iloc[i]
            cur_time = time.iloc[i]
            #---------------- strategy -----------------------
            # check if the customer is a new one. If so, set current customer as initial customer and current time as initial time, while time interval is 0
            # if not, calculate the time interval by cur_time - init_time and reset cur_time as init_time
            # -------------------------------------------------
            # check if continuous customer ids are identical
            if cur_cus == previous_cus:
                # calculate the non-work days between the selected two days
                time_interval.iloc[i] = (cur_time - previous_time).days
            else:
                previous_cus = cur_cus
                previous_time = cur_time
        # merge the interval data into df and return
        df_sorted['time_interval'] = time_interval
        return df_sorted

    def _create_interval_dataset(self, dataset, look_back):
        """
        :param dataset: input array of time intervals
        :param look_back: each training set feature length
        :return: convert an array of values into a dataset matrix.
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            dataX.append(dataset[i:i + look_back])
            dataY.append(dataset[i + look_back])
        return np.asarray(dataX), np.asarray(dataY)


    def trainingset_generation(self, enterprise_id_list_file=None, fields=["customer_id", "enterprise_id", "price", "create_time"], outdir=".", override=False):
        """
        :param enterprise_id_list_file: enterprise those data meets the minimum requirement. If not provided, function select_enterprises will be performed
        :param outdir: output directory for generated training set file
        :param fields: column header for retrieve data
        :param override: re-generate existing files
        :return: training set files corresponding to each filtered enterprise
        Note: 1. Total transaction period should be larger than train_input_length + 1 (test_set_times)
              2. if init date is not current date, it should follow time format: yyyy-mm-dd
        """
        print "Get training dataset of each enterprise..."
        # create output dir if not exists
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # get enterprise id list
        if not enterprise_id_list_file:
            enterprise_id_list_file = outdir + "/filtered_enterprise_id.txt"
        if not os.path.exists(enterprise_id_list_file) or override:
            filter_enterprises = self._select_enterprises()
            # save filtered enterprise ids to file
            list2file(filter_enterprises, enterprise_id_list_file)
        filter_enterprises = file2list(enterprise_id_list_file)
        # get transaction df
        trans_df = next(read_df_from_mysql_db(localhost=self.localhost, username=self.username, password=self.password,
                                         dbname=self.dbname, tbname=self.trans_tbname, enter_field=self.enter_field, enterprise_id=self.enter_list, fields=fields,
                                         start_time=self.init_date.strftime("%Y-%m-%d")))
        for enterprise in filter_enterprises:
            outfile = outdir + "/" + str(enterprise) + ".csv"
            # override the existing file or not
            interval_file = outdir + "/" + str(enterprise) + ".intervals.csv"
            if os.path.exists(interval_file) and not override:
                continue
            print "Retrieving transaction data of {} from transaction table".format(enterprise)
            enter_df = trans_df[trans_df['enterprise_id'] == int(enterprise)]
            # df with interval
            df_interval = self._calculate_time_interval(enter_df)
            # remove lines with time interval is 0 (when encounter new customers)
            df_interval = df_interval.ix[df_interval.time_interval > 0, :]
            # output intervals data to file for later distribution assessment and data merging
            interval_output = df_interval.time_interval
            interval_output.to_csv(interval_file)
            if os.path.exists(outfile) and not override:
               continue
            # get customers whose transaction intervals overpass the minimum requirement: training set count + 1
            # cus_trans_count = df_interval.customer_id.value_counts().index[df_interval.customer_id.value_counts() >= self.training_set_times + 1].tolist()
            # df_interval = df_interval.ix[df_interval.customer_id.isin(cus_trans_count), :]
            print "Filtering customers whose purchase times meet the minimum threshold: {}".format(self.min_purchase_count)
            df_interval = self.check_transaction_data(df_interval, init_date=self.init_date)
            # get all unique customer_ids
            all_cus_ids = df_interval.customer_id.unique()
            df_cur_enter = pd.DataFrame()
            print "Formating the dataset..."
            for current_customer in all_cus_ids:
                dataset = df_interval.time_interval[df_interval.customer_id == current_customer]
                dataset = np.asarray(dataset)
                dataX, dataY = self._create_interval_dataset(dataset, look_back=self.train_input_length)
                X_cols = []
                for x in range(1, 1+self.train_input_length):
                    X_cols.append('X' + str(x))
                dfX = pd.DataFrame(dataX, columns=X_cols)
                dfY = pd.DataFrame(dataY, columns=['Y'])
                dfY['customer_id'] = current_customer
                dfY['enterprise_id'] = enterprise
                df_cur_cus = pd.concat((dfX, dfY), axis=1)
                df_cur_enter = pd.concat((df_cur_enter, df_cur_cus), axis=0)
            # output training dataset of current enterprise to output directory
            print "Output formated training dataset to file: {}".format(outfile)
            # reindex the output file
            df_cur_enter.index = range(len(df_cur_enter.index))
            df_cur_enter.to_csv(outfile)
        print "End generation!"

def main():
    outdir = "/home/fanzong/Desktop/RNN_prediction_2/enterprise-train.5-5"
    enter_list = [76, 88, 123]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # redirect stdout to log file
    old_stdout = sys.stdout
    logfile = open(outdir + "/message.log.txt", "w")
    print "Log message could be found in file: {}".format(logfile)
    sys.stdout = logfile
    obj_trainingSet = TrainingSetSelection(enter_list=enter_list)
    obj_trainingSet.trainingset_generation(outdir=outdir)
    # return to normal stdout
    sys.stdout = old_stdout
    logfile.close()

if __name__ == "__main__":
    main()
