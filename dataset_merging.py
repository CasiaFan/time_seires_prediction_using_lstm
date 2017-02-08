__auther__ = "Arkenstone"

from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression
from scipy.stats import ks_2samp
from preprocessing import get_ids_and_files_in_dir, percentile_remove_outlier
from shutil import copyfile
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MergeSimilarDataset():
    def __init__(self,
                 input_dir,
                 enter_id_file,
                 enter_id_range,
                 output_dir,
                 dataset_length,
                 outfile_prefix="cluster",
                 **kwargs):
        """
        :param input_dir (str): input directory containing the enterprise training data
        :param enter_id_file: file containing filtered enterprise id list for analysis
        :param enter_id_range (tuple: (a, b)): enterprise ids within this range will be analyzed and clustered
        :param output_dir(str): output directory to receive output files with clustered dfs and clustering info
        :param dataset_length: dataset length of given df
        :param kwargs: ks_pv=0.1: use KS statistics p_value to assess the distribution similarity of 2 data set.
                       qq_slope=(0.9, 1.1): use slope of fitted regression line to assess the similarity of 2 data set.
                       qq_intercept = 10: another index for assess similarity of 2 data set
                       If ks_pv surpass the threshold and qq_slope is within defined range, merge the 2 data set
        """
        self.input_dir = input_dir
        self.enter_id_range = enter_id_range
        self.output_dir = output_dir
        self.outfile_prefix = outfile_prefix
        self.enter_id_file = enter_id_file
        self.dataset_length = dataset_length
        self.qq_slope_range = kwargs.get('qq_slope_range', (0.85, 1.15))
        self.qq_intercept = kwargs.get('qq_intercept', 10)
        self.ks_pv = kwargs.get('ks_pv', 0.1)

    def qq_plot(self, df_samp, df_clu):
        """
        :param df1: interval df of enterprise a. The column name should be the enterprise id
        :param df2: interval df of enterprise b. The column name should be the enterprise id
        :return: slope, intercept and total fit error of fitted regression line
        """
        # use longer list as reference distribution
        outdir = self.output_dir + "/qq-plot"
        # make output directory if not exists
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        ref = np.asarray(df_clu)
        samp = np.asarray(df_samp)
        ref_id = df_clu.columns
        samp_id = df_samp.columns
        print "Start drawing Q-Q plot using data from sample {} and cluster {}.".format(samp_id, ref_id)
        # theoretical quantiles
        samp_pct_x = np.asarray([percentileofscore(ref, x) for x in samp])
        # sample quantiles
        samp_pct_y = np.asarray([percentileofscore(samp, x) for x in samp])
        # calculate the error from real percentiles to predicted percentiles: as same as mean squared error
        pct_error = np.sum(np.power(samp_pct_y - samp_pct_x, 2)) / (2 * len(samp_pct_x))
        # estimated linear regression model
        p = np.polyfit(samp_pct_x, samp_pct_y, 1)
        regr = LinearRegression()
        model_x = samp_pct_x.reshape(len(samp_pct_x), 1)
        model_y = samp_pct_y.reshape(len(samp_pct_y), 1)
        regr.fit(model_x, model_y)
        r2 = regr.score(model_x, model_y)
        if p[1] > 0:
            p_function = "y= {} x + {}, r-square = {}".format(p[0], p[1], r2)
        elif p[1] < 0:
            p_function = "y= {} x - {}, r-square = {}".format(p[0], -p[1], r2)
        else:
            p_function = "y= {} x, r-square = {}".format(p[0], r2)
        print "The fitted linear regression model in Q-Q plot using data from enterprises {} and cluster {} is {}".format(samp_id, ref_id, p_function)
        # plot q-q plot
        x_ticks = np.arange(0, 100, 20)
        y_ticks = np.arange(0, 100, 20)
        plt.scatter(x=samp_pct_x, y=samp_pct_y, color='blue')
        plt.xlim((0, 100))
        plt.ylim((0, 100))
        # add fit regression line
        plt.plot(samp_pct_x, regr.predict(model_x), color='red', linewidth=2)
        # add 45-degree reference line
        plt.plot([0, 100], [0, 100], linewidth=2)
        plt.text(10, 70, p_function)
        plt.xticks(x_ticks, x_ticks)
        plt.yticks(y_ticks, y_ticks)
        plt.xlabel('cluster quantiles - id: {}'.format(ref_id))
        plt.ylabel('sample quantiles - id: {}'.format(samp_id))
        plt.title('{} VS {} Q-Q plot'.format(ref_id, samp_id))
        outfile = "{}/enterprise-{}-VS-cluster-{}.qqplot.png".format(outdir, samp_id, ref_id)
        plt.savefig(outfile)
        print "Plotting Q-Q plot done! The plot is stored at {}.".format(outfile)
        plt.close()
        return p[0], p[1], pct_error

    def merge_similar_dataset(self, override=True):
        """
        Merge data set within similar range used for neural network training
        :param ks_pv=0.1: use KS statistics p_value to assess the distribution similarity of 2 data set.
        :param qq_slope_range=(0.85, 1.15): use slope of fitted regression line to assess the similarity of 2 data set.
                If ks_pv surpass the threshold and qq_slope is within defined range, merge the 2 data set
        :param override=True: override exists files
        :return: files containing clustered dfs and summary clustering info
        """
        # import intervals df within enterprise id range
        ids, train_files = get_ids_and_files_in_dir(self.input_dir, self.enter_id_range, input_file_regx="^(\d+)\.csv")
        _, itv_files = get_ids_and_files_in_dir(self.input_dir, self.enter_id_range, input_file_regx="(\d+)\.intervals.csv")
        # search output directory if cluster files exist. If not, copy first file in file list to output directory as initial cluster 1 file
        interval_dir = self.output_dir + "/interval"
        train_dir = self.output_dir + "/train"
        if not os.path.exists(interval_dir):
            os.makedirs(interval_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        # record each cluster's enterprise consists
        cluster = {}
        if override:
            # remove all content in specified directory
            print "Remove old files..."
            for x in os.listdir(interval_dir):
                os.remove(interval_dir + "/" + x)
            for y in os.listdir(train_dir):
                os.remove(train_dir + "/" + y)
        if not os.listdir(interval_dir):
            id = ids.pop(0)
            cluster['0'] = np.array([int(id)])
            itv_src_file = self.input_dir + "/" + itv_files.pop(0)
            itv_dst_file = interval_dir + "/" + self.outfile_prefix + "-0.intervals.csv"
            train_src_file = self.input_dir + "/" + train_files.pop(0)
            train_dst_file = train_dir + "/" + self.outfile_prefix + "-0.csv"
            copyfile(itv_src_file, itv_dst_file)
            copyfile(train_src_file, train_dst_file)
        # read in sample interval file
        for samp_index, samp_file in enumerate(itv_files):
            samp_file = self.input_dir + "/" + samp_file
            df_samp = pd.read_csv(samp_file, header=None)
            # remove outliers
            print "Removing outliers in sample intervals..."
            df_samp_fil = percentile_remove_outlier(df_samp, 1, 1)
            df_samp_itv = df_samp_fil.ix[:, 1]
            df_samp_itv.columns = np.array(ids[samp_index])
            samp_id = ids[samp_index]
            # record slope of sample vs cluster Q-Q plot.
            # Merge sample training data to most similar cluster (slope in qq plot closest to 1).
            # If sample is distinct from all clusters, initialize it as a new cluster
            slope_to_1 = {}
            # read in cluster file
            cluster_interval_files = os.listdir(interval_dir)
            for ref_index, ref_file in enumerate(cluster_interval_files):
                # get cluster id
                id_match = re.match(r"cluster-(\d+)\.intervals\.csv", ref_file)
                cluster_id = id_match.group(1)
                ref_file = interval_dir + "/" + ref_file
                df_clu = pd.read_csv(ref_file, header=None)
                print "Removing outliers in cluster intervals..."
                df_clu_fil = percentile_remove_outlier(df_clu, 1, 1)
                df_clu_itv = df_clu_fil.ix[:, 1]
                df_clu_itv.columns = np.array(cluster_id)
                # get slope from qq plot
                qq_slope, qq_intercept, qq_error = self.qq_plot(df_samp_itv, df_clu_itv)
                ks_sta, p_value = ks_2samp(np.asarray(df_samp_itv), np.asarray(df_clu_itv))
                print "KS test results: ks-statistic: {}; p_value: {}".format(ks_sta, p_value)
                # check current sample is similar to current cluster based on ks statistics and qq slope.
                # If so, record discrepancy error from 45-degree line: sum(y_real-y_predicted)^2/2N.
                if qq_slope >= self.qq_slope_range[0] and qq_slope <= self.qq_slope_range[1]:
                    # if p_value > ks_pv:
                    if np.abs(qq_intercept) <= self.qq_intercept:
                        slope_to_1[cluster_id] = qq_error
                    else:
                        print "Purchase interval days distribution of enterprise {} is not similar enough with cluster {} (inconsistency)".format(samp_index, ref_index)
                else:
                    print "Purchase interval days distribution of enterprise {} is not similar enough with cluster {} (slope degree)".format(
                        samp_index, ref_index)
            # If not exist a similar cluster, new one
            if not slope_to_1:
                new_cluster_id = len(cluster_interval_files)
                print "Data distribution of sample {} is apparently distinct from existed clusters. " \
                      "Assign it as a new cluster {}".format(samp_id, new_cluster_id)
                interval_src_file = "{}/{}.intervals.csv".format(self.input_dir, samp_id)
                interval_dst_file = "{}/{}-{}.intervals.csv".format(interval_dir, self.outfile_prefix, new_cluster_id)
                train_src_file = "{}/{}.csv".format(self.input_dir, samp_id)
                train_dst_file = "{}/{}-{}.csv".format(train_dir, self.outfile_prefix, new_cluster_id)
                copyfile(interval_src_file, interval_dst_file)
                copyfile(train_src_file, train_dst_file)
                # store cluster info
                cluster[str(new_cluster_id)] = np.array([int(samp_id)])
            else:
                # get the most similar cluster id. Merge current enterprise to cluster with minimum qq-error
                cluster_id = min(slope_to_1, key=slope_to_1.get)
                print "Data distribution of sample {} is most similar to cluster {}. " \
                      "Merge sample data with cluster data.".format(samp_id, cluster_id)
                interval_dst_file = "{}/{}-{}.intervals.csv".format(interval_dir, self.outfile_prefix, cluster_id)
                train_dst_file = "{}/{}-{}.csv".format(train_dir, self.outfile_prefix, cluster_id)
                # merge interval files
                df_clu_int = pd.read_csv(interval_dst_file, header=None)
                df_clu_int = pd.concat([df_clu_int, df_samp], axis=0)
                df_clu_int.to_csv(interval_dst_file, index=False)
                # merge train files
                df_train_samp = pd.read_csv(self.input_dir + "/" + train_files[samp_index])
                df_clu_train = pd.read_csv(train_dst_file)
                df_clu_train = pd.concat([df_clu_train, df_train_samp], axis=0)
                df_clu_train.to_csv(train_dst_file, index=False)
                # record enterprise ids in each clusters
                if cluster_id in cluster.keys():
                    cluster[cluster_id] = np.append(cluster[cluster_id], samp_id)
                else:
                    cluster[cluster_id] = np.array([samp_id])
        # output cluster info
        print "Cluster Info: {}".format(cluster)
        df_cluster = pd.DataFrame.from_dict(cluster, orient='index')
        df_cluster = pd.DataFrame.sort_index(df_cluster)
        df_cluster = df_cluster.transpose()
        cluster_info_file = self.output_dir + "/cluster-consists-info.csv"
        df_cluster.to_csv(cluster_info_file)
        print "Output each cluster consists (enterprise ids) to file: {}.".format(cluster_info_file)


    @staticmethod
    def merge_selected_dataset(clustering_ref_file, input_dir=".", infile_suffix="csv", outdir=".", outfile_prefix="merge"):
        """
        :param indir=".": input directory for training set files
        :param infile_suffix="csv": read in file in such format
        :param outdir=".": output files for merged data set files
        :param outfile_prefix="merge": prefix for output files
        :param clustering_fref_file: file containing the clustering group info
        :return: merged files
        """
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        df_ref = pd.read_csv(clustering_ref_file)
        # get cluster id list (only digits)
        cluster_list = [x for x in df_ref.columns if re.match(r"^\d+$", x)]
        for cluster_index, cluster_name in enumerate(cluster_list):
            # dfs for storing training set in same clusters
            df_train = pd.DataFrame()
            # get ids for each enterprise
            for enter_id in df_ref.ix[:, cluster_index].dropna():  ## drop na
                enter_file = input_dir + "/" + str(int(enter_id)) + "." + infile_suffix
                # check if file exists
                if os.path.exists(enter_file):
                    df_enter = pd.read_csv(enter_file)
                    df_train = pd.concat([df_train, df_enter], axis=0)
                else:
                    print "Enterprise data file not exists."
            # write cluster df to outdir
            outfile = outdir + "/" + outfile_prefix + "-" + str(cluster_name) + ".csv"
            df_train.to_csv(outfile)


def main():
    output_dir = "/home/fanzong/Desktop/RNN_prediction_2/cluster"
    cluster_file = output_dir + "/cluster-consists-info.csv"
    input_dir = "/home/fanzong/Desktop/RNN_prediction_2/enterprise-train.5-5"  # generated by trainingset_selection script
    # check if cluster file exists
    if not os.path.exists(cluster_file):
        enter_id_file = input_dir + "/filtered_enterprise_id_list.csv"
        enter_id_range = (0, 150)
        outfile_prefix = 'cluster'
        override = False
        dataset_length = 4
        ks_pv = 0.1
        qq_slope_range = (0.9, 1.1)
        qq_intercept = 10
        obj_merge = MergeSimilarDataset(input_dir=input_dir,
                                        enter_id_file=enter_id_file,
                                        enter_id_range=enter_id_range,
                                        output_dir=output_dir,
                                        dataset_length=dataset_length,
                                        outfile_prefix=outfile_prefix,
                                        qq_slope_range=qq_slope_range,
                                        qq_intercept=qq_intercept)

        # record processing message to log file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        stdout_backup = sys.stdout
        log_file = output_dir + "/clustering_input_data.log.txt"
        log_file_handler = open(log_file, 'w')
        print "Message during clustering enterprise data could be found in {}".format(log_file)
        sys.stdout = log_file_handler
        # check if input data file exists
        if not os.listdir(input_dir):
            raise ValueError("Input files are not found in {}. Check input directory,".format(input_dir))
        # similarity assessment index
        print "Start clustering input enterprise dataset from {} to {}...".format(enter_id_range[0], enter_id_range[1])
        obj_merge.merge_similar_dataset(override=override)
        print "Clustering done! You could find them in {}".format(output_dir)
        log_file_handler.close()
        sys.stdout = stdout_backup
    else:
        MergeSimilarDataset.merge_selected_dataset(cluster_file, input_dir=input_dir, outdir=output_dir)


if __name__ == '__main__':
    main()