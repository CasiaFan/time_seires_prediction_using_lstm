__author__ = "Arkenstone"

# from sklearn.preprocessing import MinMaxScaler
from trainingset_selection import trainingSetSelection
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
from scipy.stats import binned_statistic
from process_functions import *
import os
import sys
import random
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self,
                 training_set_dir,
                 model_save_dir,
                 prediction_save_dir,
                 prediction_file_prefix='prediction',
                 model_file_prefix='model',
                 training_set_id_range=(0, np.Inf),
                 training_set_length=3,
                 train_test_ratio=4,
                 output_dir=".",
                 **kwargs):
        """
        :param training_set_dir: directory contains the training set files. File format: 76.csv
        :param model_save_dir: directory to receive trained model and model weights. File format: model-76.json/model-weight-76.h5
        :param prediction_save_dir: directory to receive predictions. File format: prediction-76.csv
        :param prediction_file_prefix='prediction': file prefix for prediction file
        :param model_file_prefix='model': file prefix for model file
        :param training_set_range=(0, np.Inf): enterprise ids in this range (a, b) would be analyzed. PS: a must be less than b
        :param training_set_length=3: first kth columns in training set file will be used as training set and the following one is expected value
        :param train_test_ratio=3: the ratio of training set size to test set size when splitting input data
        :param output_dir=".": output directory for prediction files
        :param **kwargs: output_dim=4: output dimension of LSTM layer; activation_lstm='tanh': activation function for LSTM layers; activation_dense='relu': activation function for Dense layer; activation_last='sigmoid': activation function for last layer; drop_out=0.2: fraction of input units to drop; np_epoch=10, the number of epoches to train the model. epoch is one forward pass and one backward pass of all the training examples; batch_size=32: number of samples per gradient update. The higher the batch size, the more memory space you'll need; loss='mean_square_error': loss function; optimizer='rmsprop'
        """
        self.training_set_dir = training_set_dir
        self.model_save_dir = model_save_dir
        self.prediction_save_dir = prediction_save_dir
        self.prediction_file_prefix = prediction_file_prefix
        self.model_file_prefix = model_file_prefix
        self.training_set_id_range = training_set_id_range
        self.training_set_length = training_set_length
        self.train_test_ratio = train_test_ratio
        self.output_dir = output_dir
        self.output_dim = kwargs.get('output_dim', 8)
        self.activation_lstm = kwargs.get('activation_lstm', 'tanh')
        self.activation_dense = kwargs.get('activation_dense', 'linear')
        self.activation_last = kwargs.get('activation_last', 'linear')    # softmax for multiple output
        self.dense_layer = kwargs.get('dense_layer', 2)     # at least 2 layers
        self.lstm_layer = kwargs.get('lstm_layer', 2)
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.loss = kwargs.get('loss', 'mean_squared_error')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')

    def NN_model(self, trainX, trainY, testX, testY, model_path, model_weight_path=None, lstm_layer=True):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data set
        :param testY: epect value of test data
        :param model_path: file to store the trained model
        :param model_weight_path=None: file to store weights of trained model
        :param lstm_layer=True: add lstm layer or not
        :return: model after training
        """
        if lstm_layer:
            input_dim = trainX[1].shape[1]
            print "Training model is LSTM network!"
        else:
            input_dim = trainX.shape[1]
            print "Training model is fully-connected neural network!"
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfitting
        if lstm_layer:
        # model.add(LSTM(output_dim=self.output_dim * 2, input_dim=input_dim, activation=self.activation_lstm, return_sequences=True, dropout_U=self.drop_out))
        # model.add(LSTM(output_dim=self.output_dim * 2, input_dim=self.output_dim * 2, activation=self.activation_lstm, return_sequences=True, dropout_U=self.drop_out))
            model.add(LSTM(output_dim=self.output_dim,
                           input_dim=input_dim,
                           activation=self.activation_lstm,
                           dropout_U=self.drop_out,
                           return_sequences=True))
            for i in range(self.lstm_layer-1):
                model.add(LSTM(output_dim=self.output_dim,
                           input_dim=self.output_dim,
                           activation=self.activation_lstm,
                           dropout_U=self.drop_out,
                           return_sequences=True))
        # applying a full connected NN to accept output from LSTM layer and output 1 dim
        else:
            model.add(Dense(output_dim=self.output_dim,
                            input_dim=input_dim,
                            activation=self.activation_dense))
            model.add(Dropout(self.drop_out))
            for i in range(self.dense_layer-2):
                model.add(Dense(output_dim=self.output_dim,
                            input_dim=self.output_dim,
                            activation=self.activation_dense))
                model.add(Dropout(self.drop_out))
        model.add(Dense(output_dim=1,
                        input_dim=self.output_dim,
                        activation=self.activation_last))

        # configure the learning process
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # train the model with fixed number of epoches
        model.fit(x=trainX, y=trainY, nb_epoch=self.nb_epoch, batch_size=self.batch_size, validation_data=(testX, testY))
        # store model to json file
        model_json = model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        # store model weights to hdf5 file
        if model_weight_path:
            if os.path.exists(model_weight_path):
                os.remove(model_weight_path)
            model.save_weights(model_weight_path) # eg: model_weight.h5
        return model

    def NN_prediction(self, input_file_regx="^(\d+)\.csv", model_path=None, override=False, scaler=1, nn_model='lstm'):
        """
        :param model_path=None: RNN model path. If NOT provided, run LSTM_model function to generate one
        :param override=Fasle: rerun the model prediction no matter if the expected output file exists
        :param scaler=1: scale data set using - 1: MinMaxScaler, 2: NormalDistributionScaler
        :param nn_model='lstm': NN model chosen for training and prediction.
                            'lstm' represents for LSTM model; 'dense' represents for fullly-connected neural network model
        :return: model file, model weights files, prediction file, discrepancy statistic bar plot file
        """
        # make output directory if NOT exits
        print "%s: Initializing output directories ..." %str(dt.datetime.now())
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.prediction_save_dir):
            os.makedirs(self.prediction_save_dir)
        print "%s: Done!" %str(dt.datetime.now())
        # get training sets for RNN training
        print "%s: Scanning files within select id range ..." % str(dt.datetime.now())
        ids, files = get_ids_and_files_within_given_range(inputdir=self.training_set_dir,
                                                          range=self.training_set_id_range,
                                                          input_file_regx=input_file_regx)
        print "%s: Scanning done! Selected enterprise ids are %s" % (str(dt.datetime.now()), ','.join(str(i) for i in ids))
        if not files:
            raise ValueError("No files selected in current id range. Please check the input training set directory, "
                             "input enterprise id range or file format which should be '[0-9]+.csv'")

        # training one model to each training set
        for id_index, id_file in enumerate(files):
            # store prediction result to prediction directory
            enter_file = self.training_set_dir + "/" + id_file
            train_file = self.prediction_save_dir + "/" + self.prediction_file_prefix + "-" + str(ids[id_index]) + ".train.csv"
            test_file = self.prediction_save_dir + "/" + self.prediction_file_prefix + "-" + str(ids[id_index]) + ".test.csv"
            cus_file = self.prediction_save_dir + "/" + self.prediction_file_prefix + "-" + str(ids[id_index]) + ".customer.csv"
            # check if prediction result file exists
            if not os.path.exists(train_file) or override:
                print "%s: Processing dataset - enterprise_id is: %s" %(str(dt.datetime.now()), ids[id_index])
                print "Reading from file %s" %enter_file
                df = pd.read_csv(enter_file)
                df.index = range(len(df.index))
                # retrieve training X and Y columns
                select_col = ['customer_id']
                select_col = np.append(select_col, ['X' + str(i) for i in range(1, 1+self.training_set_length)])
                select_col = np.append(select_col, ['Y', 'enterprise_id'])
                df_selected = df.ix[:, select_col]
                # remove outlier records
                df_selected = percentile_remove_outlier(df_selected, self.training_set_length)
                print df_selected.head()
                # scale the input df
                print "Scaling..."
                try:
                    if scaler == 1:
                        df_scale, minVal, maxVal = MinMaxScaler(df_selected, self.training_set_length)
                    if scaler == 2:
                        df_scale, meanVal, stdVal = NormalDistributionScaler(df_selected, self.training_set_length)
                except:
                    raise ValueError("scaler must be 1 or 2!")
                # df_scale, minVal, maxVal = df, 0, 1
                print df_scale.head()
                # split df into training set and test set
                print "Randomly selecting training set and test set..."
                training_size = int(len(df_scale.index) * self.train_test_ratio/(self.train_test_ratio + 1))
                test_size = len(df_scale.index) - training_size
                # randomly select training data set
                training_set_list = random.sample(range(len(df_scale.index)), training_size)
                test_set_list = [i for i in df_scale.index if i not in training_set_list]
                # training set for LSTM of Teras is np.array type
                df_train = df_scale.ix[training_set_list, :]
                df_test = df_scale.ix[test_set_list, :]
                # generate prediction data set: training set last 2 + expected value
                df_predict = df_scale.drop_duplicates(subset=['customer_id'], keep='last')

                #############################################################################
                # sort trainX in ascending order
                df_train = df_train.apply(lambda x: np.sort(x), axis=1)
                df_test = df_test.apply(lambda x: np.sort(x), axis=1)
                df_predict = df_predict.apply(lambda x: np.sort(x), axis=1)
                #############################################################################

                # format training set data
                if nn_model is 'lstm':
                    trainX = np.asarray(df_train.ix[:, 0:self.training_set_length]).reshape((len(df_train.index), 1, self.training_set_length))
                    testX = np.asarray(df_test.ix[:, 0:self.training_set_length]).reshape((len(df_test.index), 1, self.training_set_length))
                    predictionData = np.asarray(df_predict.ix[:, 1:1+self.training_set_length]).reshape((len(df_predict.index), 1, self.training_set_length))
                    lstm_layer = True
                elif nn_model is 'dense':
                    trainX = np.asarray(df_train.ix[:, 0:self.training_set_length])
                    testX = np.asarray(df_test.ix[:, 0:self.training_set_length])
                    predictionData = np.asarray(df_predict.ix[:, 1:1+self.training_set_length])
                    lstm_layer = False
                else:
                    raise ValueError("Parameter nn_model must be 'lstm' or 'dense'! Check the input argument!")
                trainY = np.asarray(df_train.ix[:, self.training_set_length])
                testY = np.asarray(df_test.ix[:, self.training_set_length])
                # store dfs in original scale
                df_train_init = df_selected.ix[training_set_list, :]
                df_train_init.index = range(len(df_train_init.index))
                df_test_init = df_selected.ix[test_set_list, :]
                df_test_init.index = range(len(df_test_init.index))
                df_predict_init = df_selected.drop_duplicates(subset=['customer_id'], keep='last')
                df_predict_init.index = range(len(df_predict_init))
                # create and fit the NN model
                model_path = self.model_save_dir + "/" + self.model_file_prefix + "-" + str(ids[id_index]) + ".json"
                model_weight_path = self.model_save_dir + "/" + self.model_file_prefix + "-weight-" + str(ids[id_index]) + ".h5"
                # if model not provided, run NN to train one
                if not os.path.exists(model_path) or override:
                    print "Current parameters of model: " \
                          "input training set dim: %d; " \
                          "output dim of NN: %d; " \
                          "activation function: %s; " \
                          "dropout rate from LSTM to fully connected NN: %f; " \
                          "loss function of NN: %s; " \
                          "activation function of NN: %s;" \
                          "batch_size: %d: " \
                          "number of epoches: %d" % (
                          self.training_set_length, self.output_dim, self.activation_lstm, self.drop_out, self.loss, self.activation_dense,
                          self.batch_size, self.nb_epoch)
                    print "%s: Model training start!" % str(dt.datetime.now())
                    model = self.NN_model(trainX, trainY, testX, testY, model_path, model_weight_path, lstm_layer=lstm_layer)
                    print "%s: Training finished!" % str(dt.datetime.now())
                    print "Trained model is stored in this file: %s" % model_path
                    print "Trained model weights is stored in this file: %s" % model_weight_path
                    print "%s, Save model to disk finished!" % str(dt.datetime.now())
                else:
                    # load model json string
                    print "%s: Model exists! Retrieving it from %s" %(str(dt.datetime.now()), model_path)
                    json_file = open(model_path, 'r')
                    model_json = json_file.read()
                    json_file.close()
                    model = model_from_json(model_json)
                    # load model weight
                    model.load_weights(model_weight_path)
                    model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
                    print "Model load finished!"
                # estimate model performance - Computes the loss on input data, batch by batch. Return a list of scalars
                print "%s: Evaluating error of each epoch during model training of training set and test set..." %str(dt.datetime.now())
                train_score = model.evaluate(trainX, trainY, batch_size=self.batch_size)
                test_score = model.evaluate(testX, testY, batch_size=self.batch_size)
                # rescale train_score to initial scale
                if scaler == 1:
                    init_train_score = np.asarray(train_score) * (maxVal - minVal) + minVal
                    init_test_score = np.asarray(test_score) * (maxVal - minVal) + minVal
                else:
                    init_train_score = np.asarray(train_score) * stdVal + meanVal
                    init_test_score = np.asarray(test_score) * stdVal + meanVal
                print "Train score: %s" %str(init_train_score)
                print "Test score: %s" %str(init_test_score)
                # generate prediction for training
                print "%s: Predicting the output of training set, test set and real time set..." %str(dt.datetime.now())
                train_prediction = model.predict(trainX, batch_size=self.batch_size)
                test_prediction = model.predict(testX, batch_size=self.batch_size)
                cus_prediction = model.predict(predictionData, batch_size=self.batch_size)
                # rescale to initial scale
                if scaler == 1:
                    train_prediction_init = train_prediction * (maxVal - minVal) + minVal
                    test_prediction_init = test_prediction * (maxVal - minVal) + minVal
                    cus_prediction_init = cus_prediction * (maxVal - minVal) + minVal
                else:
                    train_prediction_init = train_prediction * stdVal + meanVal
                    test_prediction_init = test_prediction * stdVal + meanVal
                    cus_prediction_init = cus_prediction * stdVal + meanVal
                cus_proba_predict = model.predict_proba(predictionData, batch_size=self.batch_size)
                print "%s: Output prediction results to directory %s" %(str(dt.datetime.now()), self.prediction_save_dir)
                df_train_init['prediction'] = pd.DataFrame(train_prediction_init)
                df_test_init['prediction'] = pd.DataFrame(test_prediction_init)
                df_predict_init[['prediction', 'probability']] = pd.DataFrame(np.concatenate([cus_prediction_init, cus_proba_predict]).reshape(2, len(cus_prediction_init)).transpose())
                # output train prediction file
                df_train_init.to_csv(train_file)
                df_test_init.to_csv(test_file)
                df_predict_init.to_csv(cus_file)

                # statistic of discrepancy between expected value and real value
                print "%s: Statistics of discrepancy between predicted and real value and presentation with bar plot..." %str(dt.datetime.now())
                discrepancyTrainRatio = np.abs(trainY - train_prediction.ravel() / trainY)
                discrepancyTestRatio = np.abs(testY - test_prediction.ravel() / testY)
                discrepancyTrain = (trainY - train_prediction.ravel()) * stdVal + meanVal
                discrepancyTest = (testY - test_prediction.ravel()) * stdVal + meanVal
                discrepancyRatio = np.concatenate([discrepancyTrainRatio, discrepancyTestRatio])
                discrepancy = np.concatenate([discrepancyTrain, discrepancyTest])
                df_discrepancy = pd.DataFrame(discrepancy, columns=['dis'])
                dis_detail = df_discrepancy.dis.apply(lambda x: round(x, 2)).value_counts().sort_index(ascending=True)
                # statistic binned value
                bins1 = [0.0, 0.1, 0.3, 0.6, 1, 2, 5, np.inf]
                bins2 = [0.0, 2.0, 5.0, 10.0, 15.0, 30, np.inf]
                statistic_ratio, bin_edge_ratio, binnumber_ratio = binned_statistic(x=discrepancyRatio,
                                                                                    values=discrepancyRatio,
                                                                                    statistic='count',
                                                                                    bins=bins1)
                statistic, bin_edge, binnumber = binned_statistic(x=discrepancy,
                                                                  values=discrepancy,
                                                                  statistic='count',
                                                                  bins=bins2)
                # bar plot for discrepancy
                X_index_ratio = np.arange(len(statistic_ratio), dtype=float)
                X_index = np.arange(len(statistic), dtype=float)
                print "Start plotting subplot1..."
                plt.subplot(2, 1, 1)
                plt.bar(X_index_ratio,
                        statistic_ratio,
                        color='b')
                plt.xlabel('|x_real - x_expected|/x_real')
                plt.ylabel('count')
                plt.title('discrepancy ratio between predicted and next real time interval')
                for x, y in zip(X_index_ratio, statistic_ratio):
                    plt.text(x=x+0.4, y=y+0.05, s=str(y), ha='center', va='bottom', fontweight='bold')
                plt.xticks(X_index_ratio, bins1[0:-1])
                # bar plot for discrepancy ratio
                print "Start plotting subplot2..."
                plt.subplot(2, 1, 2)
                plt.bar(X_index,
                        statistic,
                        color='b')
                plt.xlabel('|x_real - x_expected|')
                plt.ylabel('count')
                plt.title('discrepancy between predicted and next real time interval')
                for x, y in zip(X_index, statistic):
                    plt.text(x=x + 0.4, y=y + 0.05, s=str(y), ha='center', va='bottom', fontweight='bold')
                plt.xticks(X_index, bins2[0:-1])
                # output figure
                plot_dir = self.prediction_save_dir + "/statistic"
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig_name = plot_dir + "/statistic-" + ids[id_index] + ".png"
                plt.savefig(fig_name)
                plt.close()
                statistic_data_file_name = plot_dir + "/statistic-" + ids[id_index] + ".csv"
                dis_detail.to_csv(statistic_data_file_name)
                print "%s: Statistics and plotting enterprise %s done!" %(str(dt.datetime.now()), ids[id_index])

if __name__ == "__main__":
    nn_model = 'dense'
    output_dir = "path-to-your-output-directory"
    training_set_id_range = (9, 9)
    training_set_length = 5
    dense_layer = 4
    prediction_file_prefix = 'prediction'
    model_file_prefix = 'model'
    training_set_dir = output_dir
    model_save_dir = output_dir + "/" + model_file_prefix
    prediction_save_dir = output_dir + "/" + prediction_file_prefix
    # training_set_regx_format = "cluster-(\d+)\.csv"
    training_set_regx_format = "merge-(\d+)\.csv"
    obj_NN = NeuralNetwork(output_dir=output_dir,
                           training_set_dir=training_set_dir,
                           model_save_dir=model_save_dir,
                           prediction_save_dir=prediction_save_dir,
                           model_file_prefix=model_file_prefix,
                           prediction_file_prefix=prediction_file_prefix,
                           training_set_id_range=training_set_id_range,
                           training_set_length=training_set_length,
                           dense_layer=dense_layer)

    # record program process printout in log file
    stdout_backup = sys.stdout
    log_file_path = output_dir+"/NN_model_running_log.txt"
    log_file_handler = open(log_file_path, "w")
    print "Log message could be found in file: %s" % log_file_path
    sys.stdout = log_file_handler
    # check if the training set directory is empty. If so, run the training set selection
    if not os.listdir(obj_NN.training_set_dir):
        print "Training set files not exist! Run trainingSetSelection.trainingSetGeneration to generate them! "
        print "%s : Start running generating training set files..." % str(dt.datetime.now)
        trainingSetObj = trainingSetSelection(training_set_times_range=(20, np.inf))
        trainingSetObj = trainingSetSelection(training_set_times_range=(20, np.inf))
        trainingSetObj.trainingSetGeneration(outdir=obj_NN.training_set_dir)
        print "%s : Training set file generation done! They are store at %s directory!" %(str(
            dt.datetime.now()), obj_NN.training_set_dir)

    # predict using LSTM model
    print "%s: Prediction using NN model start!" %str(dt.datetime.now())
    obj_NN.NN_prediction(input_file_regx=training_set_regx_format, override=True, scaler=2, nn_model=nn_model)
    print "%s: Prediction done!" %str(dt.datetime.now())
    print "Models and their parameters are stored in %s; Prediction results and plot of statistic results are stored in %s" %(obj_NN.model_save_dir, obj_NN.prediction_save_dir)
    # close log file
    log_file_handler.close()
    sys.stdout = stdout_backup
