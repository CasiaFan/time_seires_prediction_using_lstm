__author__ = "Arkenstone"

import re
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy.stats import binned_statistic


def read_df_from_mysql_db(localhost, username, password, dbname, tbname, fields, enterprise_id=None, chunksize=None, enter_field=None, time_field=None, start_time=None, end_time=None):
    """ We use pd.read_sql as df extraction tools and pymysql is the default mysql connect engine
    :param tbname: table name in mysql. It could a string or a list of string
    :param fields: fields in mysql table that will be extracted. It could be a string or a list of string.
    :param enterprise_id: If specified, only data of these enterprises will be retrieved.
    :param chunksize: if this parameter is specified, then function will return an generator yielding df with chunk size rows each time
    :param enter_field: field in mysql define the the enterprise id
    :param time_field: field in mysql table define the time field. Necessary when enterprise_id is specified.
    :param start_time: used defined time period range. Only data within it will be extracted
    """
    connect_string = "mysql+pymysql://{}:{}@{}/{}".format(username, password, localhost, dbname)
    con = create_engine(connect_string)
    time_cond = ""
    if time_field:
        if not end_time:
            time_cond = " WHERE " + time_field + " <= NOW()"
        else:
            time_cond = " WHERE " + time_field + " <= '" + end_time + "'"
        if start_time:
            time_cond += " AND " + time_field + " >= '" + start_time + "'"
    enter_cond = ""
    if enterprise_id:
        sign = " AND "
        if not time_cond:
            sign = " WHERE "
        if isinstance(enterprise_id, str) or isinstance(enterprise_id, int):
            enter_cond += sign + enter_field + " = " + str(enterprise_id)
        elif isinstance(enterprise_id, tuple):
            enter_cond += sign + enter_field + " IN " + str(enterprise_id)
        elif isinstance(enterprise_id, list):
            enter_cond += sign + enter_field + " IN (" + ",".join([str(i) for i in enterprise_id]) + ")"
        else:
            raise ValueError("Argument enterprise only accept str, int, tuple or list type! Please check you input!")
    if isinstance(tbname, str):
        if isinstance(fields, str):
            fields = [fields]
        sql_cmd = "SELECT " + ",".join(fields) + " FROM " + tbname + time_cond + enter_cond
        if chunksize:
            for cur_df in pd.read_sql(sql_cmd, con, chunksize=chunksize):
                cur_df.columns = fields
                yield cur_df
        else:
            cur_df = pd.read_sql(sql_cmd, con)
            cur_df.columns = fields
            yield cur_df
    elif isinstance(tbname, list):
        if isinstance(fields, str):
            fields = [fields]
        for cur_tb in tbname:
            sql_cmd = "SELECT " + ",".join(fields) + " FROM " + cur_tb + time_cond + enter_cond
            if chunksize:
                for cur_df in pd.read_sql(sql_cmd, con, chunksize=chunksize):
                    cur_df.columns = fields
                    yield cur_df
            else:
                cur_df = pd.read_sql(sql_cmd, con)
                cur_df.columns = fields
                yield cur_df
    else:
        print "Argument tbname only accept a string or a list of string! But input type is {}".format(type(tbname))
        exit(-1)


def file2list(file):
    # read in file to list
    data = []
    with open(file, 'rb') as fi:
        for line in fi:
            if line.strip():
                data.append(line.strip())
    fi.close()
    return data


def list2file(data, file):
    # write list into given file
    with open(file, 'wb') as fo:
        fo.write("\n".join([str(i) for i in data]))
    fo.close()


def percentile_remove_outlier(df, filter_start, filter_end):
    # remove outlier records according to quantile outlier theory
    filter_region = df.ix[:, filter_start:filter_end+1]
    transaction_interval_array = np.asarray(filter_region).ravel()
    q1 = np.percentile(transaction_interval_array, 25)
    q3 = np.percentile(transaction_interval_array, 75)
    outlier_low = q1 - (q3 - q1) * 1.5
    outlier_high = q3 + (q3 - q1) * 1.5
    # make sure every element in the filtering region within normal range
    df_fil = df.ix[
        ((filter_region > outlier_low) & (filter_region < outlier_high)).sum(axis=1) == (filter_end - filter_start) + 1,]
    # re-index the filtering df from 0
    df_fil.index = range(len(df_fil.index))
    print "Outlier removed! Low boundary: {}, high boundary: {}".format(outlier_low, outlier_high)
    return df_fil


def MinMaxScaler(df, start_col_index, end_col_index):
    # MinMax scale the training set columns of df, return scaled df and Min & Max value
    maxValue = np.asarray(df.ix[:, start_col_index:end_col_index+1]).ravel().max()
    minValue = np.asarray(df.ix[:, start_col_index:end_col_index+1]).ravel().min()
    df.ix[:, start_col_index:end_col_index+1] = df.ix[:, start_col_index:end_col_index+1].apply(
        lambda x: (x - minValue) / (maxValue - minValue))
    return df, minValue, maxValue


def NormalDistributionScaler(df, start_col_index, end_col_index):
    # scale data set to normal distribution, return scaled df and mean & std value
    mean = np.asarray(df.ix[:, start_col_index:end_col_index]).ravel().mean()
    std = np.asarray(df.ix[:, start_col_index:end_col_index]).ravel().std()
    df.ix[:, start_col_index:end_col_index] = df.ix[:, start_col_index:end_col_index].apply(
        lambda x: (x - mean) / std)
    return df, mean, std


def get_ids_and_files_in_dir(inputdir, range, input_file_regx="^(\d+)\.csv"):
    """
    :param range: tuple like (0, 100)
    :param input_file_regx: input file format for regular expression
    :return: enterprise ids and file paths list
    """
    ids, files = [], []
    for file in os.listdir(inputdir):
        pattern_match = re.match(input_file_regx, file)
        if pattern_match:
            current_id = pattern_match.group(1)
            if int(current_id) >= range[0] and int(current_id) <= range[1]:
                ids.append(current_id)
                files.append(file)
    return ids, files


def binning_date_y(df, y_col=-1, n_group=5):
    """
    binning date into given numbers of groups
    :param y_col: column index of y in df. Default is the last one
    :param n_group: number of bin groups
    """
    date_y = np.asarray(df.ix[:, y_col])
    bin_percentile_width = 100 / n_group
    # get bin boundary list. To make every group have similar number of samples, we use percentiles to determine the delimiter between bins
    bin_boundary = [0]
    for i in range(n_group):
        boundary_value = np.percentile(date_y, bin_percentile_width * (i + 1))
        bin_boundary.append(boundary_value)
    bin_date_y = binned_statistic(date_y, date_y, bins=bin_boundary)[2]
    df.ix[:, y_col] = bin_date_y
    return df, bin_boundary


