__author__ = "Arkenstone"

import datetime as dt
import pandas as pd
import numpy as np
import random

def get_not_work_days(df):
    # get dates those have none transactions
    # input: df with customer_id, transaction_time
    # output: not_work_days list

    # remove hour, minute, second and microsecond info to estimate time in days
    df_time = df['create_time'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    time_list = df_time.tolist()
    # get total working time duration
    min_time = np.min(time_list)
    max_time = np.max(time_list)
    # get all day list in between the min transaction time and max transaction time
    day_list = pd.date_range(start=min_time, end=max_time)
    # get all work days without redundancy
    work_day_list = pd.unique(time_list)
    work_day_list = np.sort(work_day_list)
    # get not work days
    non_work_day_list = [x for x in day_list if x not in work_day_list]
    return non_work_day_list

def get_recency_age(df_time_interval, recency= True, age=True, remove_not_work_days=False):
    # get days from last work day to last personal purchase day. If remove_not_work_days = true, not include the days without any transactions.
    # input: df with customer_id, transaction_time
    # output: df with recency

    # remove hour, minute, second and microsecond info to estimate time in days
    df_time = df_time_interval['create_time'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    df_time_interval['create_time'] = df_time
    # get last work day
    last_work_day = df_time.max()
    # get non-work days
    non_work_day_list = get_not_work_days(df_time_interval)
    # get last date of each customer
    dict_last_purchase = df_time_interval.set_index('customer_id')['create_time'].to_dict()
    # get first date of each customer
    # df_time_reverse = df_time_interval.sort_values(['customer_id', 'create_time'], ascending=[True, False])
    df_time_reverse = df_time_interval.drop_duplicates(['customer_id'], keep='first')
    dict_first_purchase = df_time_reverse.set_index('customer_id')['create_time'].to_dict()
    dict_rec = {}
    dict_age = {}
    # get recency and age
    for i in dict_last_purchase:
        if remove_not_work_days:
            non_work_day_len_last = len([x for x in non_work_day_list if x > dict_last_purchase[i] and x <= last_work_day])
            dict_rec[i] = (last_work_day - dict_last_purchase[i]).days - non_work_day_len_last
            non_work_day_len_first = len([x for x in non_work_day_list if x > dict_first_purchase[i] and x <= last_work_day])
            dict_age[i] = (last_work_day - dict_first_purchase[i]).days - non_work_day_len_first
        else:
            dict_rec[i] = (last_work_day - dict_last_purchase[i]).days
            dict_age[i] = (last_work_day - dict_first_purchase[i]).days

    # convert dict to df
    df_recency = pd.DataFrame.from_dict(dict_rec, orient="index")
    df_recency.columns = ['recency']
    df_age = pd.DataFrame.from_dict(dict_age, orient="index")
    df_age.columns = ['age']
    if recency and age:
        return df_recency, df_age
    elif recency:
        return df_recency
    elif age:
        return df_age
    else:
        raise ValueError("At least one argument of recency and age should be True, since return cannot be none!")

def calculate_time_interval(df, customer_average=False, transaction_count=False, transaction_amount=False, remove_not_work_days=False):
    # input df: customer_id, transaction_time; output df: customer_id, transaction_time, intervals
    # input should be sorted by 1. customer 2. time in ascending order
    # if customer_average is true, then calculate the customers average transaction intervals and need 2 dfs to accept the output
    # so as when transaction_count is true.
    df_sorted = df.sort_values(['customer_id', 'create_time'], ascending=[True, True])
    cus = df_sorted['customer_id']
    # Only use day time
    time = df_sorted['create_time'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0))
    price = df_sorted['price']
    # iloc function is to get element from dataframe
    previous_cus = cus.iloc[0]
    previous_time = time.iloc[0]
    init_time = time.iloc[0]
    # initialize interval pd series
    time_interval = pd.DataFrame(np.zeros(len(cus)))
    time_interval.index = df_sorted.index
    # initial a dictionary for holding average freq
    cus_ave_freq = {}
    cus_tras_count = {}
    cus_tras_amount = {}
    cus_tras_count[previous_cus] = 1
    cus_tras_amount[previous_cus] = price.iloc[0]
    # get the non-work days
    if remove_not_work_days:
        non_work_days_list = get_not_work_days(df_sorted)
    for i in range(1, len(cus)):
        cur_cus = cus.iloc[i]
        cur_time = time.iloc[i]
        cur_price = price.iloc[i]
        # check if the customer is a new one. If so, set current customer as initial customer and current time as initial time, while time interval is 0
        # if not calculate the time interval by cur_time - init_time and reset cur_time as init_time
        # check if continuous customer ids are identical
        if cur_cus == previous_cus:
            # calculate the non-work days between the selected two days
            if remove_not_work_days:
                # non-work days between continuous two transactions
                non_work_days_len = len([x for x in non_work_days_list if x < cur_time and x > previous_time])
                time_interval.iloc[i] = (cur_time - previous_time).days - non_work_days_len
                # calculate the average interval
                # non-work days during customer total purchase duration
                non_work_days_all_len = len([x for x in non_work_days_list if x < cur_time and x > init_time])
                if cur_time != init_time:
                    cus_ave_freq[cur_cus] = ((cur_time - init_time).days - non_work_days_all_len) / cus_tras_count[cur_cus]
            else:
                time_interval.iloc[i] = (cur_time - previous_time).days
                # just in case the last two transactions occured at same day
                if cur_time != previous_time:
                    cus_ave_freq[cur_cus] = (cur_time - init_time).days / cus_tras_count[cur_cus]
            ########## Note: merge all transactions happen in one day as one ################
            if (cur_time - previous_time).days >= 1:
                cus_tras_count[cur_cus] += 1.
            # cumulate customer's transaction amount
            cus_tras_amount[cur_cus] += cur_price
            previous_time = cur_time
        else:
            previous_cus = cur_cus
            previous_time = cur_time
            init_time = cur_time
            cus_tras_count[cur_cus] = 1.
            cus_tras_amount[cur_cus] = cur_price
    # merge the interval data into df and return
    df_sorted['time_interval'] = time_interval
    # print time_interval
    # convert average dictionary to data frame
    df_ave_count = pd.DataFrame.from_dict(cus_ave_freq, orient="index")
    df_ave_count.columns = ['personal_average_time_interval']
    df_count = pd.DataFrame.from_dict(cus_tras_count, orient="index")
    df_count.columns = ['purchase_times']
    df_amount = pd.DataFrame.from_dict(cus_tras_amount, orient="index")
    df_amount.columns = ['total_purchase_amount']
    if customer_average and transaction_count and transaction_amount:
        return df_sorted, df_ave_count, df_count, df_amount
    elif customer_average and transaction_amount:
        return df_sorted, df_ave_count, df_amount
    elif transaction_count and transaction_amount:
        return df_sorted, df_count, df_amount
    elif customer_average and transaction_count:
        return df_sorted, df_ave_count, df_count
    elif customer_average:
        return df_sorted, df_ave_count
    else:
        return df_sorted

def customer_selection(df, cus_number, selection_type):
    ######################## Note: all customers in rows should be unique #################
    # input: df for selection, cus_number: count of customers to be selected; selection_type: based on frequency or recency: 1: frequency, 2: recency
    # algorithm: sort selection type value in ascendancy; select top 5% as max A part; divide the rest into 3 parts by identical value interval - min, medium, max B part; choose cus_number/3 from ecah part

    # sort type:
    if selection_type is 1:
        df_sort_type = df['purchase_times']
        df_sort_type.index = df['customer_id']
    elif selection_type is 2:
        df_sort_type = df['not_purchase_days']
        df_sort_type = df['customer_id']
    else:
        raise ValueError("selection_type must be 1 or 2")
    # remove nan in the df
    df_sort_type = df_sort_type.dropna()
    df = df.ix[df_sort_type.index, :]
    # get the min value and 95th max value in the list
    list_min = df_sort_type.min()
    list_95th = np.percentile(np.asarray(df_sort_type), 95)
    list_interval = (list_95th - list_min) / 3.
    # set mark for min, medium, max part
    mark1 = list_min + list_interval
    mark2 = list_min + 2 * list_interval
    # get df of each part
    df_min = df[df_sort_type <= mark1]
    df_medium = df[(df_sort_type <= mark2) & (df_sort_type > mark1)]
    df_max = df[df_sort_type > mark2]
    # randomly select cus_number / 3 customers from each part
    # normal situation: sample size from each part is same;
    # but if populations of max part is less than origin sample size,
    # choose all population and extra size will be compensate by medium part; if medium is in the same situation with max part,
    # use min part as candidate
    normal_sample_size = int(round(cus_number / 3.))
    min_sample_size, medium_sample_size, max_sample_size = [normal_sample_size for n in range(3)]
    if len(df_max.index) <= max_sample_size:
        max_sample_size = len(df_max.index)
        medium_sample_size = medium_sample_size + normal_sample_size - len(df_max.index)
    if len(df_medium.index) <= medium_sample_size:
        min_sample_size = min_sample_size + medium_sample_size - len(df_medium.index)
        medium_sample_size = len(df_medium.index)
    if len(df_min.index) <= normal_sample_size:
        # raise ValueError("Error: no enough samples. Please check the total sample size or change the selection sample size")
        min_sample_size = len(df_min.index)
        print "Potential ERROR with sample size input: there are not enough population compared with the sample size you defined"
    df_ran_min = df_min.ix[random.sample(df_min.index, min_sample_size), :]
    df_ran_medium = df_medium.ix[random.sample(df_medium.index, medium_sample_size), :]
    df_ran_max = df_max.ix[random.sample(df_max.index, max_sample_size), :]
    # concatenate the dfs to single one (row by row)
    df_ran_total = pd.concat([df_ran_min, df_ran_medium, df_ran_max], axis=0)
    return df_ran_total

def get_up_and_down_line_of_potential_churn(df_time_interval, up=75, down=25, adjust_factor=1, auto_adjust = False):
    ########### Note: we will choose 25th transaction interval as down line and 75th one as up line.################
    # If personal average transaction interval is less than the down line, we will use the down line as there purchase time point,
    # just in case the extreme low ridiculous purchase time point due to highly frequent consumptions. So as to the high line.
    # However, if the the low line is still too low or too high, we will use up/down regulate the cutoff line by multiplication of a adjustment factor.
    # Normally, this factor is often determined by the industry where enterprise is in.
    # input: df containing the intervals of all transaction intervals, adjustment factor, boolean auto_adjust.
    # If auto_adjust parameter is true, adjust_factor loaded will be overlooked
    # output: up line, down line
    df_time_interval = df_time_interval['time_interval']
    # remove those interval is 0
    df_no_zero = df_time_interval[df_time_interval > 0]
    # get the down and up cutoff
    cutoff_down = np.percentile(np.asarray(df_no_zero), down)
    cutoff_up = np.percentile(np.asarray(df_no_zero), up)
    # auto adjust the adjust_factor based on the 25th low line if the auto_adjust parameter is true
    if auto_adjust:
        if cutoff_down <= 1.5:
            adjust_factor = 4
        elif cutoff_down <= 2.5 and cutoff_down > 1.5:
            adjust_factor = 3
        elif cutoff_down <= 3.5 and cutoff_down > 2.5:
            adjust_factor = 2
        elif cutoff_down <= 4.5 and cutoff_down > 3.5:
            adjust_factor = 1.5
        else:
            adjust_factor = 1
    else:
        adjust_factor = adjust_factor
    print "adjustment factor for this enterprise is %d" % adjust_factor
    print "enterprise low outline is %f and high outline is %f" % (adjust_factor * cutoff_down, adjust_factor * cutoff_up)
    return adjust_factor * cutoff_down, adjust_factor * cutoff_up

def merge_time_intervals(df_time_interval):
    # input: df with customer_id and time intervals
    # output: df with time interval list -- eg: 1,2,1,5,28

    # drop na in the interval column
    df_dropna = df_time_interval.dropna(subset=['time_interval'])
    my_intervals = {}
    for i in range(len(df_dropna.index)):
        cur_cus = df_dropna.at[i, 'customer_id']
        cur_time_interval = df_dropna.at[i, 'time_interval']
        # remove the intervals = 0 in the list for continuous transactions occured in one day are treated as single on
        # warning: this also will remove customers with only one transaction
        if cur_time_interval != 0:
            if cur_cus in my_intervals:
                my_intervals[cur_cus] += ',' + str(cur_time_interval)
            else:
                my_intervals[cur_cus] = str(cur_time_interval)
    # return interval list dfs
    df_list = pd.DataFrame.from_dict(my_intervals, orient="index")
    # restore all customers with more than 1 transactions
    df_out = pd.DataFrame(np.nan, index=pd.unique(df_time_interval.customer_id), columns=['time_interval_list'])
    df_out.time_interval_list = df_list
    return df_out

def get_kth_interval_list(df_time_interval_list, k):
    # input: df with time_interval_list. eg: 1,2,1,5,28
    # output: list containing all first k transaction intervals
    # drop na in the time_interval_list column
    df_filter = df_time_interval_list.dropna(subset=['time_interval_list'])
    # get df with time_intervals as value and customer-id as index (customer_id should be input df's index)
    df_interval = df_filter.set_index(df_filter.index)['time_interval_list'].apply(lambda x: x.split(','))
    # a. test if intervals is less than k-1: if so, all intervals will be used; if not, only first k-1 transaction intervals will be used
    # df_interval_k = df_interval.apply(lambda x: x if len(x) <= k - 1 else x[0:k])
    # b. get kth intervals from each personal transaction history whose transaction count is larger than k
    interval_k_list =[df_interval[j][k-1] for j in df_interval.index if len(df_interval[j]) >= k]
    return interval_k_list

def get_deviation_interval_with_mean_sd(mean, sd, factor):
    # input: mean, sd df or value
    #  output: get deviation interval from mean based on sd and fold change of mean/sd
    deviation = sd * np.log2(1 + mean / sd) * factor
    return deviation

def get_mean_sd_max_of_transactions(df_time_interval_list, filter=False, group=False, k=5):
    # get mean and df from personal transactions; If group is true, get mean and sd of all first k (from 2 to 5) transactions; If filter=True, return mean and sd after removing the outliers
    # and sd as group reference for later churn classification
    # input: df with time_interval_list
    # output: df with personal mean and sd and max interval removed as outlier; if group is true, return group mean and sd

    # drop 0 in the time_interval_list column for nan is replaced by 0
    df_filter = df_time_interval_list[df_time_interval_list.time_interval_list != 0]
    # get df with time_intervals as value and customer-id as index (customer_id should be input df's index)
    df_interval = df_filter.set_index(df_filter.index)['time_interval_list'].apply(lambda x: x.split(','))
    # get personal mean and sd
    df_mean = df_interval.apply(lambda x: np.mean(map(float, x)))
    df_sd = df_interval.apply(lambda x: np.std(map(float, x)))
    df_max = df_interval.apply(lambda x: np.max(map(float, x)))
    # replace those sd less than 1 with 1 for their fold is required
    df_sd[df_sd < 1] = 1
    df_fil = pd.concat([df_mean, df_sd, df_max], axis=1)
    df_fil.columns = ['personal_mean', 'personal_sd', 'personal_max']
    # filter the outliers if filter is true
    if filter:
        df_fil['removed_max_interval'] = np.nan
        df_fil['remaining_max_interval'] = np.nan
        df_fil['outline_high'] = df_fil.personal_mean + get_deviation_interval_with_mean_sd(df_fil.personal_mean, df_fil.personal_sd, 1.25)
        df_fil['outline_low'] = df_fil.personal_mean - get_deviation_interval_with_mean_sd(df_fil.personal_mean, df_fil.personal_sd, 1.25)
        for i in df_interval.index:
            interval_list_fil = [x for x in df_interval[i] if float(x) >= df_fil['outline_low'][i] and float(x) <= df_fil['outline_high'][i]]
            df_fil['personal_mean'][i] = np.mean(map(float, interval_list_fil))
            df_fil['personal_sd'][i] = np.std(map(float, interval_list_fil))
            # get max interval of personal history
            interval_removed = [x for x in df_interval[i] if float(x) > df_fil['outline_high'][i]]
            if interval_removed:
                df_fil['removed_max_interval'][i] = np.max(map(float, interval_removed))
            # get max interval after removing outliers
            interval_after_remove = [x for x in df_interval[i] if float(x) <= df_fil['outline_high'][i]]
            df_fil['remaining_max_interval'][i] = np.max(map(float, interval_after_remove))

    # get first k group mean and sd
    if group:
        # get df with only kth intervals
        interval_k_list = get_kth_interval_list(df_filter, k)
        # calculate the mean and sd of total first k
        k_mean = np.mean(map(float, interval_k_list))
        k_sd = np.std(map(float, interval_k_list))
        if filter:
            interval_k_list_fil = [x for x in interval_k_list if float(x) > k_mean - get_deviation_interval_with_mean_sd(k_mean, k_sd, 1) and float(x) < k_mean + get_deviation_interval_with_mean_sd(k_mean, k_sd, 1)]
            k_mean = np.mean(map(float, interval_k_list_fil))
            k_sd = np.std(map(float, interval_k_list_fil))
            return k_mean, k_sd
        else:
            return k_mean, k_sd
    else:
        if filter:
            return df_fil[['personal_mean', 'personal_sd','personal_max', 'removed_max_interval', 'remaining_max_interval']]
        else:
            return df_fil[['personal_mean', 'personal_sd', 'personal_max']]

def get_churn_cutoff(df_time_interval_list, high_outline, low_outline, churn_cutoff = 60, filter=True, largest_k=5, auto_adjust_churn_factor=False):
    # input: df with time_interval_list; filter: if true, return mean and sd after filtering outliers;
    # largest_k: get group mean and sd until kth transaction; auto_adjust_churn_factor: if true, churn and potential churn boundary will be multiplied by a factor based on its sd value
    # output: df with personal/group churn and potential churn thresholds

    # fill na in time_interval_list with 0
    df_time_interval_list = df_time_interval_list.fillna(0)
    # get first k group mean, sd and interval list: k from 2 to 5
    k_mean = {}
    k_sd = {}
    k_churn_adjust_factor = {}
    for k in range(2, largest_k+1):
        # get group first k mean and sd after filtering outliers
        k_mean[k], k_sd[k] = get_mean_sd_max_of_transactions(df_time_interval_list, filter=filter, group=True, k=k)
        # adjust sd if auto_adjust_churn_factor is true
        # set up churn factor for later churn and potential churn calculation
        k_churn_adjust_factor[k] = 1
        if auto_adjust_churn_factor:
            # if k_sd[k] < 3 and k_sd[k] >= 2:
            #    k_churn_adjust_factor[k] = 1.5
            if k_sd[k] < 2.5 and k_sd[k] >= 1.5:
                k_churn_adjust_factor[k] = 1.5
            if k_sd[k] < 1.5:
                k_churn_adjust_factor[k] = 2
    # get personal mean and sd after filtering outliers
    df_personal_fil = get_mean_sd_max_of_transactions(df_time_interval_list, filter=filter, group=False)
    # check if the removed max interval is less than high outline. If so, set true
    df_personal_fil['removed_max_interval_larger_than_high_outline'] = (df_personal_fil.removed_max_interval.dropna() >= high_outline)
    #### Note: for we need to use fold of mean/sd, sd shouldn't be 0. We will replace those less 1 with 1 for ease of later calculation ####
    df_personal_fil.personal_sd[df_personal_fil.personal_sd < 1] = 1
    # adjust sd if auto_adjust_churn_factor is true:
    df_personal_fil['churn_factor'] = 1
    if auto_adjust_churn_factor:
        # df_personal_fil.churn_factor[(df_personal_fil.personal_sd < 3) & (df_personal_fil.personal_sd >= 2)] = 1.5
        df_personal_fil.churn_factor[(df_personal_fil.personal_sd < 2.5) & (df_personal_fil.personal_sd >= 1.5)] = 1.5
        df_personal_fil.churn_factor[df_personal_fil.personal_sd < 1.5] = 2
    # get personal churn and potential cutoff: multiply the sd with sqrt fold difference between mean and sd
    df_potential_churn_personal = df_personal_fil.personal_mean + get_deviation_interval_with_mean_sd(df_personal_fil.personal_mean, df_personal_fil.personal_sd, df_personal_fil.churn_factor)
    df_churn_personal = df_personal_fil.personal_mean + 2 * get_deviation_interval_with_mean_sd(df_personal_fil.personal_mean, df_personal_fil.personal_sd, df_personal_fil.churn_factor)
    # init a df to hold the cutoff data. Those without time interval list data cell will be filled with 0
    df_cutoff = pd.DataFrame(0, index=df_time_interval_list.index, columns=['potential_churn', 'churn'])
    df_cutoff.potential_churn = df_potential_churn_personal
    df_cutoff.churn = df_churn_personal
    # set standard for using personal churn threshold or group churn threshold: use sd/mean ratio to classify: if personal sd > mean, use group result, otherwise itself
    for i in df_time_interval_list.index:
        # check the transaction count to verify which group it belongs to
        personal_transaction_count = len(str(df_time_interval_list.time_interval_list[i]).split(','))
        # if only 1 to 3 transactions, use k=2 group mean and sd to calculate churn cutoff
        if personal_transaction_count == 1:
            # check if group cutoff is less than low outline or larger than high outline
            group_potential_churn_outline = k_mean[2] + get_deviation_interval_with_mean_sd(k_mean[2], k_sd[2], k_churn_adjust_factor[2])
            if group_potential_churn_outline > high_outline:
                df_cutoff.potential_churn[i] = high_outline
                # check if potential churn high line is larger than churn cuttoff. if so, use churn 2*potential_churn as its personal churn threshold
                if high_outline > churn_cutoff:
                    df_cutoff.churn[i] = 2 * high_outline
                else:
                    df_cutoff.churn[i] = k_mean[2] + 2 * get_deviation_interval_with_mean_sd(k_mean[2], k_sd[2], k_churn_adjust_factor[2])
            elif group_potential_churn_outline < low_outline:
                df_cutoff.potential_churn[i] = low_outline
                df_cutoff.churn[i] = 2 * low_outline
            # check if sd is too small compared with mean or its absolute value is too small
            else:
                mean_sd_fold = float(k_mean[2]) / float(k_sd[2])
                df_cutoff.potential_churn[i] = k_mean[2] + get_deviation_interval_with_mean_sd(k_mean[2], k_sd[2], k_churn_adjust_factor[2])
                df_cutoff.churn[i] = k_mean[2] + 2 * get_deviation_interval_with_mean_sd(k_mean[2], k_sd[2], k_churn_adjust_factor[2])
        else:
            # if group mean and sd exists
            if personal_transaction_count <= largest_k:
                # use personal transaction pattern or group pattern determined by sd/mean ratio > 0.6+0.1*(n_interval-1) (while for n = 2, ratio could be 0.6, since max/min >= 4 at this time).
                # if so, use group results instead.
                if float(df_personal_fil['personal_sd'][i]) / float(df_personal_fil['personal_mean'][i]) > 0.6 + 0.1 * (personal_transaction_count - 2):
                    # compare group potential churn with high and low outline
                    group_potential_churn_outline = k_mean[personal_transaction_count] + get_deviation_interval_with_mean_sd(k_mean[personal_transaction_count], k_sd[personal_transaction_count], k_churn_adjust_factor[personal_transaction_count])
                    if group_potential_churn_outline > high_outline:
                        df_cutoff.potential_churn[i] = high_outline
                        # check if potential churn high line is larger than churn cuttoff. if so, use churn 2*potential_churn as its personal churn threshold
                        if high_outline > churn_cutoff:
                            df_cutoff.churn[i] = 2 * high_outline
                        else:
                            df_cutoff.churn[i] = k_mean[personal_transaction_count] + 2 * get_deviation_interval_with_mean_sd(k_mean[personal_transaction_count], k_sd[personal_transaction_count], k_churn_adjust_factor[personal_transaction_count])
                    elif group_potential_churn_outline < low_outline:
                        df_cutoff.potential_churn[i] = low_outline
                        df_cutoff.churn[i] = 2 * low_outline
                    else:
                        df_cutoff.potential_churn[i] = k_mean[personal_transaction_count] + get_deviation_interval_with_mean_sd(k_mean[personal_transaction_count], k_sd[personal_transaction_count], k_churn_adjust_factor[personal_transaction_count])
                        df_cutoff.churn[i] = k_mean[personal_transaction_count] + 2 * get_deviation_interval_with_mean_sd(k_mean[personal_transaction_count], k_sd[personal_transaction_count], k_churn_adjust_factor[personal_transaction_count])
                else:
                    if df_cutoff.potential_churn[i] > high_outline:
                        df_cutoff.potential_churn[i] = high_outline
                        if high_outline > churn_cutoff:
                            df_cutoff.churn[i] = 2 * high_outline
                    if df_cutoff.potential_churn[i] < low_outline:
                        df_cutoff.potential_churn[i] = low_outline
                        df_cutoff.churn[i] = 2 * low_outline
            else:
                if df_cutoff.potential_churn[i] > high_outline:
                    df_cutoff.potential_churn[i] = high_outline
                    if high_outline > churn_cutoff:
                        df_cutoff.churn[i] = 2 * high_outline
                if df_cutoff.potential_churn[i] < low_outline:
                    df_cutoff.potential_churn[i] = low_outline
                    df_cutoff.churn[i] = 2 * low_outline
    # merge personal mean, sd and potential, churn df for output
    df_out = pd.concat([df_personal_fil, df_cutoff], axis=1)
    # print out kth mean and sd
    for i in k_mean:
        print "the %d th group mean is %f, and the group sd is %f" % (i, k_mean[i], k_sd[i])
    return df_out

def set_customer_churn_mark(df):
    # input: df with recency, time_interval_list
    # output: origin df with churn mark: active, potential_churn, churn
    df['mark'] = 'potential_churn'
    # approximate date to days
    df.mark[(np.round(df.recency) <= np.round(df.potential_churn))] = 'active'
    df.mark[(np.round(df.recency) > np.round(df.churn))] = 'churn'
    # if the recency is less than largest interval removed as outlier (which also is less than the high outline) and the mark of this customer is churn, modify this to portential churn
    df.mark[(np.round(df.recency) > np.round(df.churn)) &
            (-df.removed_max_interval_larger_than_high_outline.fillna(True)) &
            (np.round(df.recency) <= np.round(df.personal_max.fillna(0)))] = 'potential_churn'
    # if the largest interval after removing outliers is still larger than the recency while the customer is marked as churn, remarking as potential churn
    df.mark[(np.round(df.recency) > np.round(df.churn)) &
            (np.round(df.remaining_max_interval) >= np.round(df.recency))] = 'potential_churn'
    return df
