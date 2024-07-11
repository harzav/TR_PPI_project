# -*- coding: utf-8 -*-
"""
Predict the outcome based on trained model(s) finalised post ensemble feature selection - combination of feature
filter and GA based feature selection. Finalised models hold parameters optimised via GA

Uses majority voting based ensemble method for prediction. Models (from the ensemble) are chosen for each patient
guided by missing features threshold i.e. if 50% of the features used by a model is missing for a given patient
then it skips that model to be used for disease prediction of this patient.

A final list of outcomes are printed that includes - List of patients not predicted at all,
csv of prediction classes of patients and csv of voting value i.e. mean for each of the predicted class

Works for both binary-class multi-label problem and multi-class single-label problem. For multi-class multi-label
problem, change the output i.e. Y to binary-class and multi-label problem

"""

import numpy as np
import math
import logging
import csv
import joblib

import argparse
import copy

import pandas as pd

from knnimpute import knn_impute_optimistic

from joblib import Parallel, delayed
from tensorflow.keras.models import load_model


# ######## PreProcess class ###############################################
class PreProcess:

    # Constructor, called at the start of this class
    def __init__(self, dataset_filename, class_labels, percentage, neighbour, normalization_method,
                 missing_imputation_method):
        """
        Pre process constructor
        :param dataset_filename: filename of the training dataset
        :param class_labels: class labels of the dataset
        :param percentage:  default missing data percentage threshold check
        :param neighbour: default neighbours for knn impute and LOF
        :param normalization_method: default normalization is MaxMin scaling [0,1]
        :param missing_imputation_method: default is knn impute
        """
        # These values are either passed from the command line or uses the default values above

        # variables for these parameters
        self.percentage = float(percentage)
        self.neighbour = neighbour
        self.normaliser = normalization_method
        self.impute = missing_imputation_method

        self.output_message = ''
        self.pcaComponents = 0.9  # number of components to keep for PCA

        # data files and label files
        self.dataset_filename = dataset_filename
        self.classLabels = class_labels

    # Accessor functions for parameters
    def set_percentage(self, value):
        self.percentage = value

    def get_percentage(self):
        return self.percentage

    def set_neighbour(self, value):
        self.neighbour = value

    def get_neighbour(self):
        return self.neighbour

    def set_normaliser(self, value):
        self.normaliser = value

    def get_normaliser(self):
        return self.normaliser

    def set_impute(self, value):
        self.impute = value

    def get_impute(self):
        return self.impute

    @staticmethod
    def convert_to_array(dataset_filename, alpha_flag):
        """
        function to read txt datafile with features X samples and create list of lists
        adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
        Script has been edited for empty string and feature label reading + reading any non-float character as empty
        :param dataset_filename: file name of dataset
        :param alpha_flag: alpha Flag signifies if 1st column is alphanumeric or not
        :return: return list of lists dataset
        """
        # open datafile and extract content into an array, and close
        # datafile has only x variables i.e. features and does not have target i.e. y variables
        dataset = []  # initialise to empty list
        with open(dataset_filename, "r") as dataset_fid:
            number_of_lines = 0

            for line1 in dataset_fid:
                dataset.append([])  # list of lists
                words = line1.split(",")  # comma separated delimiter

                for i, word in enumerate(words):
                    if i == 0 and alpha_flag:  # alpha Flag signifies if 1st column is alphanumeric or not
                        dataset[number_of_lines].append(word)  # alphanumeric feature codes
                    else:
                        if word == "":  # check for empty string. empty string is not converted to float
                            # print(words[i])
                            dataset[number_of_lines].append(float('nan'))  # -1000 for empty string
                        else:
                            # print(words[i])
                            try:
                                dataset[number_of_lines].append(float(word))  # change string to float
                            except ValueError:
                                dataset[number_of_lines].append(float('nan'))  # -1000 for empty string

                number_of_lines = number_of_lines + 1  # increment counter to read next line in datafile

            print("Dataset file was successfully parsed! %s features lines read" % number_of_lines)

        return dataset  # return list of lists dataset

    def average_duplicate_measurements(self, dataset_initial, markers):
        """
        function to check duplicate features. If duplicate then take its average
        adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
        :param dataset_initial: initial dataset
        :param markers: lsit of duplicate features
        :return:
        """
        dataset = {}  # initialise dictionary to hold features as key and sum of its multiple observations as list value
        dict_of_occurences = {}  # initialise dictionary to hold feature as key and its occurences as value
        num_of_elements = 0  # initialise counter for features iterated
        column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
        row_count = len(dataset_initial)  # row count is features count

        for i in range(row_count):
            if markers[i] not in dataset:  # if feature not already present in dictionary then add
                dict_of_occurences[markers[i]] = 1  # counter of occurrence of feature set to 1
                dataset[markers[i]] = []  # initialise list to hold value against specific feature key

                for j in range(column_count):
                    if (dataset_initial[i][j] != '') and dataset_initial[i][j] != float("inf") \
                            and dataset_initial[i][j] != float("-inf"):
                        # exclude null values for average calculation
                        dataset[markers[i]].append(float(dataset_initial[i][j]))  # append columns to feature key
                    else:
                        dataset[markers[i]].append(0)  # append 0 for missing values

            else:
                dict_of_occurences[markers[i]] += 1  # increment the counter of occurrence

                # if feature key already exists then do column specific addition
                for j in range(column_count):
                    if (dataset_initial[i][j] != '') and dataset_initial[i][j] != float("inf") \
                            and dataset_initial[i][j] != float("-inf"):
                        # exclude null values for average calculation
                        dataset[markers[i]][j] = dataset[markers[i]][j] + float(dataset_initial[i][j])

            num_of_elements += 1  # increment counter for features iterated

        # calculate average for each feature key
        for key in dataset:  # iterate over keys
            for j in range(len(dataset[key])):
                dataset[key][j] = dataset[key][j] / dict_of_occurences[key]

        data = []  # initialise list to hold average value
        markers = []  # initialise list to hold feature names
        num_of_markers = 0

        # segregate average data and features
        for key, vals in dataset.items():
            data.append([])
            markers.append(key)
            for i in range(len(vals)):
                data[num_of_markers].append(vals[i])
            num_of_markers += 1

        self.output_message += "Features count after duplicate feature averaging %s" % len(data) + "\n"

        return [data, markers]  # return average data and features

    def perform_missing_values_imputationperform_missing_values_imputation(self, dataset_initial):
        """
        function for missing value imputation using KNN, default is k=20
        adapted from script written by Konstantinos Theofilatos
        :param dataset_initial: initial dataset
        :return: imputed dataset with missing values
        """

        missing_imputation_method = self.get_impute()  # get impute method
        neighbors = self.get_neighbour()  # get neighbors count for KNN, default is k=20
        row_count = len(dataset_initial)
        column_count = len(dataset_initial[0])
        # missing values imputation
        averages = [0] * row_count
        if missing_imputation_method == "1":  # average imputation
            num_of_non_missing_values = [0] * row_count  # initialise list for missing value count
            for j in range(row_count):
                for i in range(column_count):
                    if dataset_initial[j][i] != float("nan") and dataset_initial[j][i] != '':
                        # print(dataset_initial[j][i+1])
                        averages[j] += float(dataset_initial[j][i])
                        num_of_non_missing_values[j] += 1

                averages[j] = averages[j] / float(num_of_non_missing_values[j])

            self.output_message += 'Average imputation method was used!\n'

            for i in range(row_count):
                for j in range(column_count):
                    if dataset_initial[i][j] == float("nan") and dataset_initial[i][j] == '':
                        dataset_initial[i][j] = averages[i]

            return dataset_initial

        else:  # KNN-impute
            # convert data to observation specific from feature specific
            dataset_initial = list(map(list, zip(*dataset_initial)))
            for i in range(len(dataset_initial)):
                for j in range(len(dataset_initial[0])):
                    if dataset_initial[i][j] == '' or dataset_initial[i][j] == float("nan"):
                        dataset_initial[i][j] = np.NaN

            dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)),
                                            k=neighbors, verbose=True)
            dataset = list(map(list, zip(*dataset)))
            self.output_message += 'KNN imputation method was used!\n'
            return dataset


# ######## prediction class ###############################################

class PredictRegression:
    def __init__(self, inputx, model_list, features_list, multilabel, dict_chain, predict_data_initial,
                 feature_namelist, predict_proteins, thread_num):
        # variables for these parameters
        self.model_list = model_list
        self.inputx = inputx
        self.features_list = features_list  # model specific list of features selected
        # sequence of feature names written in front1 feature final is alphabetic
        # whereas features for dataset is not alphabetic. SO need to do name matching to read right values
        self.feature_namelist = feature_namelist
        self.predict_proteins = predict_proteins
        self.multilabel = multilabel
        self.dict_chain = dict_chain  # this is list of list holding classifier chain for each model
        self.missingFeatThreshold = 0.10
        self.predict_data_initial = np.transpose(np.array([np.array(x) for x in predict_data_initial]))
        if int(thread_num) > 2:
            self.threads = int(thread_num) - 2
        else:
            self.threads = 1

    # ## user input for ensemble or single model
    def predict_fun(self):
        # Input validation
        # ##### Load the Finalized Model from disk using joblib

        not_predicted = []  # patients not predicted
        predicted_patient = []  # patients predicted
        # row_count = self.inputx.shape[0]
        mean_class = []
        # counter to pick 1st patient predicted as patients may not be predicted at all if missing
        # features exceeds threshold
        mean_counter = 0
        
        # iterate on each patient, check for missing % of features in original data and choose the models
        # for patient in range(5,6): # for testing
        for patient in range(self.inputx.shape[0]):

            # get the value for 1st key and choose this as seq to align other outputs i.e. y
            y_chain = self.dict_chain[0]
            pred_array = []  # initialise prediction class to empty list
            counter = 0  # counter for model count
            for mdl in copy.deepcopy(self.model_list):  # iterate over map object
                # new_x = []  # initialise X of the model
                colindices = []  # feature indices selected for after normalisation and impute predict data

                if mdl.endswith(".hdf5"):
                    clf = load_model(mdl)
                    neural = True  # Flag for neural model selection
                else:
                    clf = joblib.load(mdl)
                    neural = False
                
                # get features and map inputX to shape of model i.e. features uses
                # X is of shape [n_samples, n_features]; counter maps to model list counter
                for feature in range(len(self.features_list[0])):  # iterate on features
                    if self.features_list[counter][feature] == 1:  # append only those feature that are 1 i.e. selected
                        colindices.append(feature)  # get the indices of features selected
                
                # read specific patient i.e. observation record for the feature selected
                new_x = self.inputx[[patient], :]  # get the row
                new_x = new_x[:, colindices]  # get the columns

                # get unprocessed X for selected features to do missing feature check
                unprocessed_x = self.predict_data_initial[[patient], :]  # get rows
                unprocessed_x = unprocessed_x[:, colindices]  # get teh columns
                # nan signifies missing value
                missing_count = len(unprocessed_x[np.where(unprocessed_x == float("nan"))])
                total_feature = len(colindices)
                missing_pct = float(missing_count/total_feature)

                counter = counter + 1
                if missing_pct >= self.missingFeatThreshold:
                    continue  # go to next model for prediction

                else:
            
                    if self.multilabel:  # multi label problem
                        for i, c in enumerate(clf):  # guided by classifier chain, picks the sequence of Ys
                            if i == 0:
                                try:
                                    if not neural:
                                        y_pred = (c.predict(new_x)).reshape(-1, 1)
                                    else:
                                        new_x = np.reshape(new_x, (new_x.shape[0], new_x.shape[1], 1))
                                        y_pred = np.argmax(c(new_x), axis=1).reshape(-1, 1)
                                except ValueError:
                                    y_pred = [np.array([0.])]
                            else:
                                # add the prior y or Ys predicted to X
                                input_stacked = np.hstack((new_x, y_pred))
                                try:
                                    if not neural:
                                        new_y = c.predict(
                                            input_stacked)  # predict y using the model trained for X and Y
                                    else:
                                        input_stacked = np.reshape(input_stacked,
                                                                   (input_stacked.shape[0], input_stacked.shape[1], 1))
                                        new_y = np.argmax(c(input_stacked), axis=1)
                                except ValueError:
                                    new_y = np.array([0.])
                                y_pred = np.hstack((y_pred, new_y.reshape(-1, 1)))

                        if counter == 1:  # 1st classifier chain is chosen as base, so start from 2nd
                            # list of ndarrays containing prediction per patient for each model
                            pred_array.append(y_pred)
                        else:
                            # swap y_pred guided by y_chain
                            new_list = y_pred.tolist()
                            shuffle_y = []
                            new_chain = self.dict_chain[counter - 1]
                            shuffle_y.append([])
                            for j in range(len(y_chain)):
                                for k in range(len(new_chain)):
                                    if y_chain[j] == new_chain[k]:  # compare positions
                                        shuffle_y[0].append(new_list[0][k])
                            # list of ndarrays containing prediction per patient for each model
                            pred_array.append(shuffle_y)
                    else:
                        # single label problem
                        if neural:
                            new_x = np.reshape(new_x, (new_x.shape[0], new_x.shape[1], 1))
                            p2 = clf(new_x)
                        else:
                            p2 = clf.predict(new_x)  # X is single sample
                        p2 = np.reshape(p2, (p2.shape[0]))
                        pred_array.append(p2)
                
            # ## get majority vote or mean class for each patient across multilabel
            if len(pred_array) == 0:  # i.e. pred_array = [] if all models skipped and no prediction for the patient
                not_predicted.append(patient)
                continue
            else:
                mean_counter = mean_counter + 1
                predicted_patient.append(patient)
                patient_mean = np.mean(pred_array, axis=0)

                # print(patient_mean_val)
                patient_mean[np.where(patient_mean >= 0.5)] = 1.0  # so that false negatives are minimised
                patient_mean[np.where(patient_mean < 0.5)] = 0.0

            # if patient == 0: ### use an index
            if mean_counter == 1:  # mean_counter is initialised to zero
                mean_class = patient_mean
            elif mean_counter > 1:
                mean_class = np.vstack((mean_class, patient_mean))
        if mean_counter >= 1:
            # add patients serial no to mean class so that patient ids can be retrieved from original data
            mean_class = np.hstack((mean_class, np.array(predicted_patient).reshape(-1, 1)))

        return mean_class

    def predict_fun_parallel(self):
        # Input validation
        # ##### Load the Finalized Model from disk using joblib

        row_count = self.inputx.shape[0]

        # counter to pick 1st patient predicted as patients may not be predicted at all if missing
        # features exceeds threshold

        # iterate on each patient, check for missing % of features in original data and choose the models
        # for patient in range(5,6): # for testing
        mean_class = \
            Parallel(n_jobs=self.threads, verbose=10)(delayed(self.predict_fun_thread)(patient)
                                                      for patient in range(row_count))

        # mean_class = np.hstack((mean_class, np.array(predicted_patient).reshape(-1, 1)))

        return mean_class

    def predict_fun_thread(self, patient):

        # get the value for 1st key and choose this as seq to align other outputs i.e. y
        y_chain = self.dict_chain[0]
        pred_array = []  # initialise prediction class to empty list
        counter = 0  # counter for model count
        for mdl in copy.deepcopy(self.model_list):  # iterate over map object

            colindices = []  # feature indices selected for after normalisation and impute predict data

            if mdl.endswith(".hdf5"):
                clf = load_model(mdl)
                neural = True  # Flag for neural model selection
            else:
                clf = joblib.load(mdl)
                neural = False

            # get features and map inputX to shape of model i.e. features uses
            # X is of shape [n_samples, n_features]; counter maps to model list counter
            for feature in range(len(self.features_list[counter])):  # iterate on features
                if self.features_list[counter][feature] == 1:  # append only those feature that are 1 i.e. selected
                    colindices.append(feature)  # get the indices of features selected

            # read specific patient i.e. observation record for the feature selected
            new_x = self.inputx[[patient], :]  # get the row
            new_x = new_x[:, colindices]  # get the columns

            # get unprocessed X for selected features to do missing feature check
            unprocessed_x = self.predict_data_initial[[patient], :]  # get rows
            unprocessed_x = unprocessed_x[:, colindices]  # get teh columns
            # nan signifies missing value
            missing_count = len(unprocessed_x[np.where(unprocessed_x == float("nan"))])
            missing_count_1000 = len(unprocessed_x[np.where(unprocessed_x == float(-1000.0))])
            #print(missing_count_1000)
            total_feature = len(colindices)
            #print('features total', total_feature)
            missing_pct = float(missing_count / total_feature)
            missing_pct_1000 = float(missing_count_1000 / total_feature)
            #print('missing pct', missing_pct)
            #print('missing pct 1000', missing_pct_1000)

	
            counter = counter + 1
            #print(counter)
            if (missing_pct >= self.missingFeatThreshold) or (missing_pct_1000 >= self.missingFeatThreshold):
                #print('missing pct', missing_pct)
                #print('model', mdl)
                continue  # go to next model for prediction

            else:

                if self.multilabel:  # multi label problem
                    for i, c in enumerate(clf):  # guided by classifier chain, picks the sequence of Ys
                        if i == 0:
                            try:
                                if not neural:
                                    y_pred = (c.predict(new_x)).reshape(-1, 1)
                                else:
                                    new_x = np.reshape(new_x, (new_x.shape[0], new_x.shape[1], 1))
                                    y_pred = np.argmax(c(new_x), axis=1).reshape(-1, 1)
                            except ValueError:
                                y_pred = [np.array([0.])]
                        else:
                            # add the prior y or Ys predicted to X
                            input_stacked = np.hstack((new_x, y_pred))
                            try:
                                if not neural:
                                    new_y = c.predict(input_stacked)  # predict y using the model trained for X and Y
                                else:
                                    input_stacked = np.reshape(input_stacked,
                                                               (input_stacked.shape[0], input_stacked.shape[1], 1))
                                    new_y = np.argmax(c(input_stacked), axis=1)
                            except ValueError:
                                new_y = np.array([0.])
                            y_pred = np.hstack((y_pred, new_y.reshape(-1, 1)))

                    if counter == 1:  # 1st classifier chain is chosen as base, so start from 2nd
                        # list of ndarrays containing prediction per patient for each model
                        pred_array.append(y_pred)
                    else:
                        # swap y_pred guided by y_chain
                        new_list = y_pred.tolist()
                        shuffle_y = []
                        new_chain = self.dict_chain[counter - 1]
                        shuffle_y.append([])
                        for j in range(len(y_chain)):
                            for k in range(len(new_chain)):
                                if y_chain[j] == new_chain[k]:  # compare positions
                                    shuffle_y[0].append(new_list[0][k])
                        # list of ndarrays containing prediction per patient for each model
                        pred_array.append(shuffle_y)
                else:
                    # single label problem
                    if neural:
                        new_x = np.reshape(new_x, (new_x.shape[0], new_x.shape[1], 1))
                        p2 = clf(new_x)

                    else:
                        p2 = clf.predict(new_x)  # X is single sample
                    p2 = np.reshape(p2, (p2.shape[0]))
                    pred_array.append(p2[0])

        # ## get majority vote or mean class for each patient across multilabel (change this calculation for each type
        # of prediction)
        if len(pred_array) == 0:  # i.e. pred_array = [] if all models skipped and no prediction for the patient
            return None, patient
        else:
            patient_mean = np.mean(pred_array, axis=0)

        mean_class = patient_mean

        return mean_class


def read_arguments():
    # Get parameters based on command line input 
    parser = argparse.ArgumentParser()
    # default type is str
    parser.add_argument('file', metavar='DATA_FILENAME', help='txt file name with omics data')
    parser.add_argument('models', metavar='TRAINED_MODELS', help='List of Pickle file names of trained models')
    parser.add_argument('features', metavar='FEATURES_TRAINED_MODELS',
                        help='CSV file name with features in binary format for the list of trained models')
    parser.add_argument('maxValFile', metavar='MAX_VAL_DATA', help='CSV file name with Max normalised value for each '
                                                                   'protein i.e. feature derived from training data')
    parser.add_argument('minValFile', metavar='MIN_VAL_DATA', help='CSV file name with Min normalised value for each '
                                                                   'protein i.e. feature derived from training data')
    parser.add_argument('orgNormData', metavar='Original_NORM_DATA', help='CSV file name with original training data '
                                                                          'normalised for KNN Impute')
    parser.add_argument('classLabelOrg', metavar='Original_Class_Labels', help='Txt name with original training data '
                                                                               'class labels for association rule')
    parser.add_argument('chain', metavar='CLASSIFIER_CHAIN', help='csv file name with classifier chain for the trained '
                                                                  'models')
    parser.add_argument('trainFeatures', metavar='TRAINING_FEATURES', help='csv file name with list of features used in'
                                                                           ' the training data. To be used for aligning'
                                                                           ' shape of prediction sample')
    
    parser.add_argument('-p', '--percentage', metavar='MISS_PERCENTAGE', nargs='?', type=float, default=0.97,
                        help='percentage missing data threshold')
    parser.add_argument('-k', '--neighbour', metavar='KNN_LOF_NEIGHBOURS', nargs='?', type=int, default=20,
                        help='neighbours for knn impute and LOF')
    parser.add_argument('-n', '--normalization_method', nargs='?', default='1',
                        help='normalisation method, default is MinMax scaling')
    parser.add_argument('--missing_imputation_method', nargs='?', default='2',
                        help='missing data impute method, default is knn')
    
    args = parser.parse_args()  # get the mandatory and default arguments
    
    return args


# function to read dictionary
def read_dict(file_name):
    chain_dict = []	
    with open(file_name, 'r') as f:
        readline = csv.reader(f)
        for i, row in enumerate(readline):  # read each row to create list of lists
            chain_dict.append([])
            chain_dict[i].extend(row)
    
    return chain_dict 


# function to read csv cells as float
def read_csv_to_float(file_name, user='unknown', jobid=0, pid=0):
    data = []	
    with open(file_name, 'r') as f:
        read = csv.reader(f)
        for row in read:  # convert yo float
            for cell in row:
                try:
                    data.append(float(cell))
                except ValueError:
                    logging.exception(
                        "PID:{}\tJOB:{}\tUSER:{}\tError in csv reader {} not convertable to float."
                        .format(pid, jobid, user, cell))
                    # print("%s not convertable to float" % cell)
    return data 


# function to pick features of predict data as per training features	
def filter_features_predict(dataset_initial, features_training):
    new_data = []  # list to hold dataset after picking features present in training
    selected = 0  # index for append in new_data
    new_proteins = []  # initialise list to hold feature names after deletion
    column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations

    # retrieve list of features from predict data
    predict_feature = []
    for j in range(len(dataset_initial)):  # row is features in predict data
        predict_feature.append(dataset_initial[j][0])  # 1st value in list is feature

    for i, feature in enumerate(features_training):  # match each feature in predict with training
        flag_found = 0  # flag to check if training feature was found in predict feature list
        for j in range(len(dataset_initial)):  # get the row index of predict data to be picked for matching feature
            if feature == dataset_initial[j][0]:  # if feature names match
                flag_found = 1
                new_data.append([])  # list of list to hold dataset after matching features
                for k in range(1, column_count):
                    new_data[selected].append(dataset_initial[j][k])
                selected += 1
                new_proteins.append(feature)  # 1st value in list is feature

        # check if training features not in predict then add null value
        if flag_found == 0:
            print("Training feature not found in predict. Adding null value for %s" % features_training[i])
            new_data.append([])  # list of list to hold dataset after matching features
            for k in range(1, column_count):
                new_data[selected].append(float("nan"))
            selected += 1
            new_proteins.append(feature)  # training feature as not found in predict data

    print('Features successfully matched with training features')

    return [new_data, new_proteins]


# function to normalize data [0,1] or logarithmic
def normalize_dataset(dataset_initial, maximums, minimums, normalization_method):
    column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
    row_count = len(dataset_initial)  # row count is features count
    if normalization_method == '1':  # arithmetic max-min sample-wise normalization
        # do min-max normalisation for each feature
        for i in range(row_count):  # row has proteins
            # print("normalising for row %s"%i)
            for j in range(column_count):  # column has observations
                if dataset_initial[i][j] != "" and dataset_initial[i][j] != float("nan"):
                    # print("normalising for column %s"%j)
                    dataset_initial[i][j] = 0 + (1/((maximums[i])-(minimums[i]))) * \
                                            (float(dataset_initial[i][j])-(minimums[i]))

        print('Arithmetic normalization was used')
        return dataset_initial
    
    else:
        logged_data = []
        for i in range(len(dataset_initial)):
            logged_data.append([])
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] == '' or dataset_initial[i][j] == float("nan"):
                    logged_data[i].append('')
                else:
                    if dataset_initial[i][j] == 0:
                        logged_data[i].append(0)
                    else:
                        logged_data[i].append(math.log2(dataset_initial[i][j]))
        
        print('Logarithmic normalization was used')
        return [logged_data]

# function to check duplicate features for original prediction data without normalisation and impute
# adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos


def average_dup_feat_predict(dataset_initial, markers):
    dataset = {}  # initialise dictionary to hold features as key and sum of its multiple observations as list value
    dict_of_occurences = {}  # initialise dictionary to hold feature as key and its occurences as value
    num_of_elements = 0  # initialise counter for features iterated
    column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
    row_count = len(dataset_initial)  # row count is features count
        
    for i in range(row_count):
        if markers[i] not in dataset:  # if feature not already present in dictionary then add
            dict_of_occurences[markers[i]] = 1  # counter of occurence of feature set to 1
            dataset[markers[i]] = []  # initialise list to hold value against specific feature key
            for j in range(column_count):
                # exclude null values for average calculation
                if dataset_initial[i][j] != float("nan") and dataset_initial[i][j] != '':
                    dataset[markers[i]].append(float(dataset_initial[i][j]))  # append columns to feature key
                else:
                    dataset[markers[i]].append(float("nan"))  # append float("nan") (and not zero) for missing values
                        
        else:
            dict_of_occurences[markers[i]] += 1  # increment the counter of occurence
                
            # if feature key already exists then do column specific addition
            for j in range(column_count):
                # exclude null values for average calculation
                if dataset_initial[i][j] != float("nan") and dataset_initial[i][j] != '':
                    dataset[markers[i]][j] = dataset[markers[i]][j] + float(dataset_initial[i][j])
        
            num_of_elements += 1  # increment counter for features iterated
    
    # calculate average for each feature key
    for key in dataset:  # iterate over keys
        for j in range(len(dataset[key])):
            dataset[key][j] = dataset[key][j]/dict_of_occurences[key]
    
    data = []  # initialise list to hold average value
    markers = []  # initialise list to hold feature names
    num_of_markers = 0
        
    # segregate average data and features
    for key, vals in dataset.items():
        data.append([])
        markers.append(key)
        for i in range(len(vals)):
            data[num_of_markers].append(vals[i])
        num_of_markers += 1
        
    return [data, markers]  # return average data and features


def predictor(dataset_initial, features, model_list, model_feature_file, classification_chain, thread_num,
              user, jobid, pid):
    """
    Predict Regression labels for a given dataset using the provided model list
    :param dataset_initial: input dataset for prediction
    :param features: input dataset's features names
    :param model_list: list of model files used for prediction
    :param model_feature_file: file with the selected features per model in the list
    :param classification_chain: file with the classification chain
    :param thread_num: number of available threads, used for parallel processes
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: the predicted data
    """

    # ##### Make Predictions on New Data
    # get original prediction data with feature matching and dup avg for missing val threshold check
    # don't change missing val i.e. -1000 to zero
    try:
        data_org_pred, proteins = average_dup_feat_predict(copy.deepcopy(dataset_initial), features)
        data_tran = np.transpose(dataset_initial)
        features_df = pd.read_csv(model_feature_file)
        # get the classifier chain
        dict_chain = read_dict(classification_chain)
        prediction = PredictRegression(data_tran, model_list, features_df.values.tolist(), False, dict_chain,
                                       data_org_pred, list(features_df.columns), features, thread_num)
        predicted_class = prediction.predict_fun_parallel()
        return [True, predicted_class]
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn error occurred during the prediction step.".format(pid, jobid,
                                                                                                          user))
        return [False, 'An error occurred during the prediction step.']


# ################## main called when predict_cvd.py is run
def main(filename, var_dict, thread_num, user='unknown', jobid=0, pid=0):
    """
    Predict labels for ncRnas from their 58 calculated features
    :param filename: input file with unaligned reads and their features
    :param var_dict: dictionary with parameters and model creation files
    :param thread_num: threads available
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: predicted labels for each ncRNA
    """

    # create instance of class preProcess with parameters
    process = PreProcess(filename, '', var_dict['percentage'], var_dict['neighbour'],
                         var_dict['normalization_method'], var_dict['missing_imputation_method'])

    # True is for alphaNumeric flag and False is for multi-label as this is X (not Y) read
    predict_data_initial = process.convert_to_array(filename, True)

    # read the features of training data
    training_features = []
    df = pd.read_csv(var_dict['train_features'], header=None)
    df_list = [item for sublist in df.values.tolist() for item in sublist]  # read flat list from dataframe
    for i in range(len(df_list)):
        training_features.append(df_list[i])

    data_new, proteins = filter_features_predict(predict_data_initial, training_features)

    # ######## use min/ max saved during training and write new function
    # read max-min csvs as list, this is full list of omics before duplicate averaging
    maximums = read_csv_to_float(var_dict['max_val_file'], user, jobid, pid)
    # print (len(maximums)) # shape : proteins X samples
    minimums = read_csv_to_float(var_dict['min_val_file'], user, jobid, pid)
    # normalize data - arithmetic [0,1]
    norm_data = normalize_dataset(copy.deepcopy(data_new), maximums, minimums, var_dict['normalization_method'])

    # don't read 1st column as index
    org_norm_data = pd.read_csv(var_dict['org_norm_data'], index_col=False, header=None)

    # combined_norm = norm_data.extend(org_norm_data.values.tolist()) combine the two, with predict data at start of row
    combined_norm = pd.concat([pd.DataFrame(norm_data), org_norm_data], axis=1)

    # missing value imputation using KNN
    impute_data = process.perform_missing_values_imputation(combined_norm.values.tolist())
    # extract the columns of impute data = len(norm_data[0])
    df = pd.DataFrame(impute_data)

    len_predict_data = len(norm_data[0]) - 1
    df_postfilter = df.loc[:, 0:len_predict_data]  # slice all rows and columns = length of original data for prediction

    # check for duplicate features. If duplicate then take its average
    mean_data, mean_proteins = process.average_duplicate_measurements(df_postfilter.values.tolist(), proteins)

    # dataset as np.array and transpose. To be used by prediction classification models
    # ####### data format for multi-label class : X (n_samples, n_features); y (n_samples, n_labels)
    new_data = np.array([np.array(x) for x in mean_data])
    # new_data = np.array([np.array(x) for x in df_postfilter.values.tolist()])
    data_tran = np.transpose(new_data)

    # #### read the models names from command argument
    model_str_list = var_dict['models']
    model_list = map(str, model_str_list.strip('[]').split(','))

    # ### read final features used csv file as list
    features_df = pd.read_csv(var_dict['features'])  # has proteins in header

    # get original prediction data with feature matching and dup avg for missing val threshold check
    # don't change missing val i.e. -1000 to zero
    data_org_pred, proteins = average_dup_feat_predict(copy.deepcopy(data_new), proteins)

    # ##### Make Predictions on New Data
    # get the classifier chain
    dict_chain = read_dict(var_dict['chain'])
    predict = PredictRegression(data_tran, model_list, features_df.values.tolist(), False, dict_chain, data_org_pred,
                                list(features_df.columns), proteins, thread_num)
    predicted_class = predict.predict_fun_parallel()
    return [True, predicted_class]


if __name__ == '__main__':

    config = {'ncrnaseq.differentialexpression':
                  {'models': '[/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                             'predictor_inputs/0finalChainOfModels.pkl,/opt/backend-application/insybio-ncrnaseq/'
                             'ncrnaseq_differential_expression/predictor_inputs/1finalChainOfModels.pkl,/opt/'
                             'backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/predictor_inputs/'
                             '2finalChainOfModels.pkl,/opt/backend-application/insybio-ncrnaseq/'
                             'ncrnaseq_differential_expression/predictor_inputs/3finalChainOfModels.pkl,/opt/'
                             'backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/predictor_inputs/'
                             '4finalChainOfModels.pkl,/opt/backend-application/insybio-ncrnaseq/'
                             'ncrnaseq_differential_expression/predictor_inputs/5finalChainOfModels.pkl,/opt/'
                             'backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/predictor_inputs/'
                             '6finalChainOfModels.pkl]',
                   'features': '/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                               'predictor_inputs/features_FinalFront1.csv',
                   'percentage': '0.97',
                   'neighbour': '20',
                   'normalization_method': '1',
                   'missing_imputation_method': '1',
                   'train_features': '/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                                     'predictor_inputs/OmicsPostDupAvg.csv',
                   'max_val_file': '/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                                   'predictor_inputs/FeatureMaxNormdata.csv',
                   'min_val_file': '/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                                   'predictor_inputs/FeatureMinNormdata.csv',
                   'chain': '/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                            'predictor_inputs/classifierChain.csv',
                   'org_norm_data': '/opt/backend-application/insybio-ncrnaseq/ncrnaseq_differential_expression/'
                                    'predictor_inputs/DataPostNorm.csv'}}

    main("featsj.thanos1.txt", config['ncrnaseq.differentialexpression'], "j.thanos", 1, 1)
