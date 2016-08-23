__author__ = "Arkenstone"

import os
import re
import numpy as np

def percentile_remove_outlier(df, training_set_length):
    # remove outlier records according to quantile outlier theory
    filter_region = df.ix[:, 1:training_set_length + 2]
    transaction_interval_array = np.asarray(filter_region).ravel()
    q1 = np.percentile(transaction_interval_array, 25)
    q3 = np.percentile(transaction_interval_array, 75)
    outlier_low = q1 - (q3 - q1) * 1.5
    outlier_high = q3 + (q3 - q1) * 1.5
    df_fil = df.ix[
        ((filter_region > outlier_low) & (filter_region < outlier_high)).sum(axis=1) == training_set_length + 1,]
    df_fil.index = range(len(df_fil.index))
    print "Outlier removed! Low boundary: %f, high boundary: %f" % (outlier_low, outlier_high)
    return df_fil

def MinMaxScaler(df, training_set_length):
    # MinMax scale the training set columns of df, return scaled df and Min & Max value
    df_copy = df.copy()
    maxValue = np.asarray(df.ix[:, 1:2+training_set_length]).ravel().max()
    minValue = np.asarray(df.ix[:, 1:2+training_set_length]).ravel().min()
    df_copy.ix[:, 1:2+training_set_length] = df.ix[:, 1:2+training_set_length].apply(
        lambda x: (x - minValue) / (maxValue - minValue))
    return df_copy, minValue, maxValue

def NormalDistributionScaler(df, training_set_length):
    # scale data set to normal distribution, return scaled df and mean & std value
    df_copy = df.copy()
    mean = np.asarray(df.ix[:, 1:2+training_set_length]).ravel().mean()
    std = np.asarray(df.ix[:, 1:2+training_set_length]).ravel().std()
    df_copy.ix[:, 1:2+training_set_length] = df.ix[:, 1:2+training_set_length].apply(
        lambda x: (x - mean) / std)
    return df_copy, mean, std

def get_ids_and_files_within_given_range(inputdir, range, input_file_regx="^(\d+)\.csv", interval=False):
    """
    :param inputdir: directory containing input files
    :param range: tuple like (0, 100)
    :param input_file_regx: input file format for regular expression
    :param interval=True: return intervals files within the range
    :return: enterprise ids and file paths list
    """
    ids, train_files, itv_files = [], [], []
    for file in os.listdir(inputdir):
        pattern_match = re.match(input_file_regx, file)
        if pattern_match:
            current_id = pattern_match.group(1)
            if int(current_id) >= range[0] and int(current_id) <= range[1]:
                ids.append(current_id)
                train_files.append(file)
                cur_itv_file = str(current_id) + ".intervals.csv"
                itv_files.append(cur_itv_file)
        else:
            continue
    if interval:
        return ids, train_files, itv_files
    else:
        return ids, train_files
