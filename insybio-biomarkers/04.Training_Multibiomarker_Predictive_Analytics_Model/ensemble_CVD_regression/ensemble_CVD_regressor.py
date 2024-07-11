# -*- coding: utf-8 -*-

"""
Pre-requisite - 
- Data files to be cleaned and combined using mergeFiles.py. 
- Classes should be held as integers starting from 0 i.e. 0,1,2 etc
- installation of packages mifs and knnimpute. please see import comments for installation link and code updates,
if applicable

This is the main script file for multi-omics biomarker identification for multi-class multi label problem.
There are two main classes - preProcess and FeatureSelection. 

- preProcess class main function is biomarker_discovery_modeller() that in-turn calls other functions within preProcess.
 preProcess class
applies missing value filter, normalisation, data impute, duplicate features averaging and outlier samples deletion. 
It returns the final dataset, feature names and class labels that in-turn are used by FeatureSelection class

- FeatureSelection class main function is biomarker_feature_selection() that in-turn calls other functions within
FeatureSelection.
FeatureSelection class applies intersection of univariate feature filter and Genetic algorithm (GA) to choose final
feature(s).
It also optimises the various parameters for feature filter methods and classification models. 

The final parameters and features selected post GA based selection are passed to FinaliseAndSaveModel().
FinaliseAndSaveModel() trains the chosen parameters, feature filter, classifier method and features on full sample
and saves the model(s) for prediction

The default and no-default parameters are specified in readArguments(). Values of default parameters specified in
readArguments() can be changed via command line argument

Works for both binary-class multi-label problem and multi-class single-label problem. For multi-class multi-label
problem, change the output i.e. Y to binary-class and multi-label problem. This is important as multi-label uses
classifier chain and normalisation method used for data is [0,1]. Hence, when different outputs i.e. Y is appended to
X in the chain, the value of Y should be between [0,1]

To run the code via command prompt:
    multi-label problem # python ensemble_CVD_MultiLabel.py omics.txt AllLabels.txt
    single label problem e.g. Diabetes # python ensemble_CVD_MultiLabel.py omics.txt DiabetesLabel.txt
To run the code via spyder:
    # run ensemble_CVD_MultiLabel.py Omics.txt AllLabels.txt
    # run ensemble_CVD_MultiLabel.py Omics.txt DiabetesLabel.txt

To change default parameters such as generation, population and change sampling to False
    # python ensemble_CVD_MultiLabel.py Omics.txt AllLabels.txt -g 3 -pp 20 -s

This code has been tested in windows system.
"""
from __future__ import print_function
import os
import random
import statistics
import csv
import numpy as np
import scipy.stats as st
import time
import math
import sys
import logging
from collections import Counter

from copy import deepcopy
import argparse

import copy

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from collections import Counter  # for counting classes from labels list
from sklearn.metrics import fbeta_score, make_scorer  # to make f2 for cross_validate
from scipy.spatial import distance
from sklearn.base import clone

from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from sklearn import svm

from joblib import Parallel, delayed
import joblib
from joblib.parallel import cpu_count
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss

from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

from mlxtend.frequent_patterns import apriori  # association rule
from mlxtend.preprocessing import TransactionEncoder

import itertools  # for permutation of multi-labels
import json  # for classifier chain dictionary writing for final front1 solutions

import mifs as mifs

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

# install from: https://pypi.org/project/knnimpute/
from knnimpute import knn_impute_few_observed, knn_impute_with_argpartition, knn_impute_optimistic, knn_impute_reference

# install from: https://pypi.org/project/smogn/
import smogn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten
#from tensorflow.keras.activations import relu
from sklearn.metrics import mean_squared_error
#from tensorflow.keras.optimizers import Adam, Nadam, SGD, RMSprop
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# SEED = 10  # seed for random state
# random.seed(10)  # for same random index generation during sampling

random.seed()  # Initialize random generator seed with the current local time

# MULTI_LABELS = []  # hold the names of the labels i.e. Y such as Diabetes, Cholestrol extracted from AllLabels.txt
indices = []  # hold the indices of features who match to one feature


# ######## pre-processing class

class PreProcess:

    # Constructor, called at the start of this class
    def __init__(self, dataset_filename, classLabels, group_feat, percentage, neighbour, normalization_method,
                 missing_imputation_method):
        # percentage  - default missing data percentage threshold check
        # neighbour   - default neighbours for knn impute and LOF
        # normalization_method - default normalisation is MaxMin scaling [0,1]
        # missing_imputation_method - default is knn impute
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
        self.classLabels = classLabels

        self.group_feat = group_feat
        self.pathProcess = ''

    # Accessor functions for parameters
    def setPercentage(self, value):
        self.percentage = value

    def getPercentage(self):
        return self.percentage

    def setNeighbour(self, value):
        self.neighbour = value

    def getNeighbour(self):
        return self.neighbour

    def setNormaliser(self, value):
        self.normaliser = value

    def getNormaliser(self):
        return self.normaliser

    def setImpute(self, value):
        self.impute = value

    def getImpute(self):
        return self.impute

    def setPath(self, value):
        self.pathProcess = value

    def getPath(self):
        return self.pathProcess

    # function to read txt datafile with features X samples and create list of lists
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # Script has been edited for empty string and feature label reading + reading any non-float character as empty
    # def convertToArray(self,dataset_filename,alphaFlag):

    def convertToArray(self, dataset_filename, alphaFlag, multiLabel):
        # open datafile and extract content into an array, and close
        # datafile has only x variables i.e. features and does not have target i.e. y variables
        dataset = []  # initialise to empty list
        with open(dataset_filename, "r") as dataset_fid:
            number_of_lines = 0
            for line1 in dataset_fid:
                dataset.append([])  # list of lists
                words = line1.split("\t")  # tab separated delimiter

                for i in range(len(words)):
                    if i == 0 and alphaFlag:  # alpha Flag signifies if 1st column is alphanumeric or not
                        '''if multiLabel:  # if reading multilabel data then update global label variable
                            global MULTI_LABELS
                            MULTI_LABELS.append(words[i])
                            # dataset[number_of_lines].append(float(words[i]))
                        else:'''

                        dataset[number_of_lines].append(words[i])  # alphanumeric feature codes
                    else:

                        if words[i] == "":  # check for empty string. empty string is not converted to float
                            # print(words[i])
                            dataset[number_of_lines].append(-1000)  # -1000 for empty string
                        else:
                            # print(words[i])
                            try:
                                dataset[number_of_lines].append(float(words[i]))  # change string to float
                            except ValueError:
                                dataset[number_of_lines].append(-1000)  # -1000 for empty string
                number_of_lines = number_of_lines + 1  # increment counter to read next line in datafile

            # print("Dataset file was successfully parsed! %s features lines read " % number_of_lines)

        return dataset  # return list of lists dataset

    # function to check for percentage missing value
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    def filter_proteomics_dataset(self, dataset_initial, group_feat):
        new_data = []  # list to hold dataset after dropping features with missing values > percentage
        selected = 0  # index for append in new_data
        new_proteins = []  # initialise list to hold feature names after deletion
        missing_proteins = 0  # initialise counter for feature count with missing value > percentage
        proteins_missing_values_percentage = 0  # count for total features count with missing value
        column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
        row_count = len(dataset_initial)
        threshold = self.getPercentage()  # get threshold for missing value percentage

        global indices
        # for j in range(len(group_feat[1])):
        # if (group_feat[0][j]!=group_feat[1][j]):
        # if flag==0:

        # indices[-1].append(j)
        # #if j-1 not in indices:
        # #	indices.append(j-1)
        # else:
        # indices.append([])
        # indices[-1].append(j)
        # flag=0
        # #indices.append(j)
        # else:
        # flag=1
        # indices.sort()

        # print("These feature indices %s match to one feature " % indices)

        # for m in range(1, column_count):
        # dataset_initial[indices[0]][m]=sum(dataset_initial[n][m] for n in indices)/len(indices)

        # print(dataset_initial[indices[0]])

        # Exclude connected features as described in 1
        for i in range(row_count):  # count missing values for each feature
            missing = 0  # initialise counter for missing value count for each feature
            for j in range(1, column_count):  # exclude 1st value as its the feature name
                # print (j)
                if (dataset_initial[i][j] == '') or dataset_initial[i][j] == -1000:
                    missing += 1
                    proteins_missing_values_percentage += 1

            if (missing / float(len(dataset_initial[0]))) < threshold:  # missing % below threshold
                new_data.append([])  # list of list to hold dataset after missing value deletion
                for k in range(1, column_count):
                    # print (k)
                    new_data[selected].append(dataset_initial[i][k])
                selected += 1
                # new_proteins.append(proteins[i])
                new_proteins.append(dataset_initial[i][0])  # 1st value in list is feature
            else:
                missing_proteins += 1  # increment counter for feature count with missing value > percentage

        # print('Data successfully filtered for missing values!')
        # print('Total Number of Features = ' + str(len(set(group_feat[0]))))
        # print('Total Number of Features with missing values less than predefined threshold=' + str(selected))
        # print('Percentage of Missing Values in all Features (Proteins)
        # = ' + str(proteins_missing_values_percentage / float(row_count * column_count)))
        self.output_message += 'Total Number of Molecules=' + str(row_count) + '\n'
        self.output_message += 'Total Number of Molecules with missing values less than allowed threshold=' + str(
            selected) + '\n'
        self.output_message += 'Percentage of Missing Values in all molecules=' + str(
            proteins_missing_values_percentage / float(row_count * column_count)) + '\n'

        return [new_data, new_proteins, indices]

    # function to check duplicate features. If duplicate then take its average
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    def average_duplicate_measurements(self, dataset_initial, markers):
        dataset = {}  # initialise dictionary to hold features as key and sum of its multiple observations as list value
        dict_of_occurences = {}  # initialise dictionary to hold feature as key and its occurrences as value
        num_of_elements = 0  # initialise counter for features iterated
        column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
        row_count = len(dataset_initial)  # row count is features count
        # print (column_count)
        # print (markers[0]) # prints protein labels

        for i in range(row_count):
            if markers[i] not in dataset:  # if feature not already present in dictionary then add
                # print (markers[i])
                dict_of_occurences[markers[i]] = 1  # counter of occurence of feature set to 1
                dataset[markers[i]] = []  # initialise list to hold value against specific feature key
                # print (dataset)
                for j in range(column_count):
                    # print (dataset_initial[i][j])
                    if dataset_initial[i][j] != -1000 and dataset_initial[i][j] != '':
                        # exclude null values for average calculation
                        dataset[markers[i]].append(float(dataset_initial[i][j]))  # append columns to feature key
                    else:
                        dataset[markers[i]].append(0)  # append 0 for missing values

            else:
                dict_of_occurences[markers[i]] += 1  # increment the counter of occurence

                # if feature key already exists then do column specific addition
                for j in range(column_count):
                    if dataset_initial[i][j] != -1000 and dataset_initial[i][j] != '':
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

        # ################# for testing, delete later########################
        # df = pd.DataFrame(dataset)
        # df.to_csv(path_or_buf = path + "Dictdata.csv")
        # ###################################################################

        # segregate average data and features
        for key, vals in dataset.items():
            data.append([])
            markers.append(key)
            for i in range(len(vals)):
                data[num_of_markers].append(vals[i])
            num_of_markers += 1

        self.output_message += "Features count after duplicate feature averaging {}\n".format(len(data))
        # print("Features count after duplicate feature averaging {}".format(data))

        return [data, markers]  # return average data and features

    # function to normalize data [0,1] or logarithmic
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # script edited to store min-max value per feature to be used from training into prediction
    def normalize_dataset(self, dataset_initial):
        normalization_method = self.getNormaliser()  # get normalisation method
        column_count = len(dataset_initial[0])  # counter for iterating on a feature across observations
        row_count = len(dataset_initial)  # row count is features count

        if normalization_method == '1':  # arithmetic max-min sample-wise normalization
            # initialise max and min to find max-min values
            maximums = [-1000.0] * row_count
            minimums = [1000.0] * row_count

            # find max and min for each feature
            for i in range(row_count):
                for j in range(column_count):
                    if dataset_initial[i][j] != "" and dataset_initial[i][j] != -1000:
                        # if not null then do max-min compare
                        if float(dataset_initial[i][j]) > maximums[i]:
                            maximums[i] = float(dataset_initial[i][j])

                        if float(dataset_initial[i][j]) < minimums[i]:
                            minimums[i] = float(dataset_initial[i][j])

            # max1 = max(maximums)
            # min1 = min(minimums)
            # print('Maximum Quantity Value:' + str(max1))
            # print('Minimum Quantity Value:' + str(min1))

            # ######## store feature specific max and min for using in test data##############################

            # do min-max normalisation for each feature
            for i in range(row_count):
                if maximums[i] != minimums[i]:
                    for j in range(column_count):
                        if dataset_initial[i][j] != "" and dataset_initial[i][j] != -1000:
                            dataset_initial[i][j] = 0 + (1 / (maximums[i] - minimums[i])) * (
                                    float(dataset_initial[i][j]) - minimums[i])
                else:
                    for j in range(column_count):
                        if dataset_initial[i][j] != "" and dataset_initial[i][j] != -1000:
                            dataset_initial[i][j] = 0 + float(dataset_initial[i][j]) - minimums[i]

            self.output_message += 'Arithmetic normalization was used!\n'

            # #####################################  write min and max to csv to use with test data ####################
            df = pd.DataFrame(maximums)
            df.to_csv(path_or_buf=self.pathProcess + os.sep + "FeatureMaxNormdata.csv", index=False, header=False)

            df = pd.DataFrame(minimums)
            df.to_csv(path_or_buf=self.pathProcess + os.sep + "FeatureMinNormdata.csv", index=False, header=False)

            ##############################################################################################

            return dataset_initial

        else:
            logged_data = []
            for i in range(len(dataset_initial)):
                # print('i='+str(i))
                logged_data.append([])
                for j in range(len(dataset_initial[0])):
                    # print('j='+str(j))
                    if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                        logged_data[i].append('')
                    else:
                        if dataset_initial[i][j] == 0:
                            logged_data[i].append(0)
                        else:
                            logged_data[i].append(math.log2(dataset_initial[i][j]))

            self.output_message += 'Logarithmic normalization was used!\n'
            return [logged_data]

    # function for missing value imputation using KNN, default is k=20
    # adapted from script written by Konstantinos Theofilatos
    def perform_missing_values_imputation(self, dataset_initial):

        missing_imputation_method = self.getImpute()  # get impute method
        neighbors = self.getNeighbour()  # get neighbors count for KNN
        row_count = len(dataset_initial)
        column_count = len(dataset_initial[0])

        # missing values imputation
        averages = [0] * row_count
        if missing_imputation_method == "1":  # average imputation
            num_of_non_missing_values = [0] * row_count  # initialise list for missing value count
            for j in range(row_count):
                for i in range(column_count):
                    if dataset_initial[j][i] != -1000 and dataset_initial[j][i] != '':
                        # print(dataset_initial[j][i+1])
                        averages[j] += float(dataset_initial[j][i])
                        num_of_non_missing_values[j] += 1

                averages[j] = averages[j] / float(num_of_non_missing_values[j])

            self.output_message += 'Average imputation method was used!\n'

            for i in range(row_count):
                for j in range(column_count):
                    if dataset_initial[i][j] == -1000:
                        dataset_initial[i][j] = averages[i]

            return dataset_initial

        else:  # KNN-impute
            # convert data to observation specific from feature specific
            dataset_initial = list(map(list, zip(*dataset_initial)))
            for i in range(len(dataset_initial)):
                for j in range(len(dataset_initial[0])):
                    if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                        dataset_initial[i][j] = np.NaN
            # print(dataset_initial)

            dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)),
                                            k=neighbors)
            # print(r)
            dataset = list(map(list, zip(*dataset)))
            self.output_message += 'KNN imputation method was used!\n'
            return dataset

    # function to get unique values or classes from list exclude -1000 or '' i.e. nulls
    def uniqueClass(self, listVal, keyLabel):
        flat_list = [item for sublist in listVal for item in sublist]  # change list of lists to list
        s = set(flat_list)
        uniqueVal = set()  # initialise unique class set
        classDict = {}  # initialise clas dictionary to hold class specific values

        for i in s:  # get non null values from set
            if i != '' and i != -1000:
                uniqueVal.add(
                    str(int(i)) + keyLabel)  # add multi-label e.g. Diabetes, Cholestrol so that dict key is unique

        for i in range(len(list(uniqueVal))):
            classDict[list(uniqueVal)[i]] = 0

        return classDict

    # outlier detection using LOF on PCA data
    def outlier_detection(self, dataset, class_Labels):
        neighbors = self.getNeighbour()  # get neighbors count for LOF

        pca = PCA(n_components=self.pcaComponents, svd_solver='full')
        new_dataset = []
        num_of_samples = 0

        # get transpose data from list of lists
        for j in range(len(dataset[0])):
            new_dataset.append([])
            for i in range(len(dataset)):
                new_dataset[num_of_samples].append(float(dataset[i][j]))
            num_of_samples += 1

        dataset_new = pca.fit_transform(
            new_dataset)  # Fit the model with new_dataset and apply the dimensionality reduction on new_dataset

        clf = LocalOutlierFactor(n_neighbors=neighbors)
        y_pred = clf.fit_predict(dataset_new)  # LOF on PCA data

        # ######################## For TESTING...LOF output ##################
        df = pd.DataFrame(y_pred)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "LOFdata.csv", index=False, header=False)
        # outlier prediction

        # #############################################################################################
        # delete the outliers i.e. where y_pred = -1
        num_of_samples = 0
        data_new = []
        data_removed = []
        delet_count = 0

        # initialise dictionary to hold class labels and its count for patients detected as outliers
        label_OutlierCount = self.uniqueClass([class_Labels[0]], 1)  # initialise with 1st dictionary
        for lbl, key in enumerate([1]):  # create dictionary of classes for each label i.e. y1, y2, y3 etc
            label_OutlierCount.update(self.uniqueClass([class_Labels[lbl]], key))  # update other dictionaries
        # print (label_OutlierCount) # {'1Z_Diabetes': 0, '0Z_Diabetes': 0, '1Z_Anti.depressant': 0,
        # '0Z_Anti.depressant': 0, '0Z_Cholesterol Therapy': 0, '1Z_Cholesterol Therapy': 0,
        # '0Z_Anti-Platelet Therapy': 0, '1Z_Anti-Platelet Therapy': 0}

        # initialise lists to hold post outlier removal labels
        if len([1]) == 1:  # for Single label
            class_LabelNew = [[]]
        else:  # for Multi label
            class_LabelNew = [[] for lbl in [1]]

        for i in range(len(dataset)):  # 419; iterate on row i.e. feature count
            data_new.append([])
            data_removed.append([])
            for j in range(len(class_Labels[0])):  # 329; iterate on column i.e. Brunneck code
                if y_pred[j] == 1:
                    data_new[num_of_samples].append(
                        float(dataset[i][j]))  # append the columns (i.e. patients) not marked as outlier
                else:
                    data_removed[num_of_samples].append(float(dataset[i][j]))

                # run this for only one feature to avoid multiple counts for same patient
                if i == 0 and y_pred[j] == 1:
                    for lbl in range(len([1])):  # get respective label observations that are not outliers
                        class_LabelNew[lbl].append(class_Labels[lbl][j])

                # run this for only one feature to avoid multiple counts for same patient
                if i == 0 and y_pred[j] == -1:
                    delet_count += 1  # increment counter for outliers; count for only 1 feature

                    for lbl, key in enumerate([1]):  # get respective patient count in outlier by class labels
                        if class_Labels[lbl][j] != '' and class_Labels[lbl][j] != -1000:
                            label_OutlierCount[str(int(class_Labels[lbl][j])) + key] += 1  # increment dict key

            num_of_samples += 1

        self.output_message += "%s" % delet_count + " observations were outliers and deleted" + "\n"

        self.output_message += "Observations after outlier deletion %s" % len(data_new[0]) + "\n"
        # self.output_message += "Label(s) for which patients in outlier detected %s" % MULTI_LABELS + "\n"
        self.output_message += "Distribution of patients in outlier detected %s" % label_OutlierCount + "\n"

        # ################################## FOR TESTING ##########################
        df = pd.DataFrame(data_removed)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "OutlierDelData.csv", index=False,
                  header=False)  # outlier prediction

        df = pd.DataFrame(class_LabelNew)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "LabelsPostOutlier.csv", index=False,
                  header=False)  # outlier prediction

        # #############################################################################################

        return data_new, class_LabelNew

    # return dataset_new

    # method for pre-processing the data file before performing ensemble feature selection
    def biomarker_discovery_modeller(self):
        # print (self.dataset_filename)

        # read the datafile and convert to array
        dataset_initial = self.convertToArray(self.dataset_filename, True, False)
        # True is for alphaNumeric flag & False for multiLabel

        # read the class labels file and convert to array; index 0 is read as string
        class_Labels = self.convertToArray(self.classLabels, True, True)  # True is for alphaNumeric + multiLabel flag
        '''class_Labels_binary = copy.deepcopy(class_Labels)
        for k in range(len(class_Labels)):
            for j in range(len(class_Labels[k])):
                if class_Labels[k][j] > 0:
                    class_Labels_binary[k][j] = 1
                else:
                    class_Labels_binary[k][j] = 0'''

        # read the file with grouped features
        # group_feat = self.convertToArray(self.group_feat, True, False)
        group_feat = []
        # print("labels read")
        # print(MULTI_LABELS, ":", len(MULTI_LABELS))
        # print (class_Labels[0])
        # print (len(class_Labels))
        # print("Observations: %d" % len(class_Labels[0]))

        # check for missing values, if % missing value beyond the threshold then drop (delete) that observation
        new_data, new_proteins, indices = self.filter_proteomics_dataset(dataset_initial, group_feat)

        # ################################### write data to csv, for testing#####################
        df = pd.DataFrame(new_data)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "DataPostMissFilter.csv", index=False, header=False)
        # data after missing value threshold deletion

        df = pd.DataFrame(new_proteins)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "OmicsPostMissFilter.csv", index=False, header=False)
        # data after missing value threshold deletion

        # normalize data - arithmetic [0,1]
        # norm_data = self.normalize_dataset(new_data)
        norm_data = copy.deepcopy(new_data)

        # #####################################  write data to csv, to b deleted later#####################
        df = pd.DataFrame(norm_data)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "DataPostNorm.csv", index=False, header=False)
        # data after normalisation
        #################################################################################################

        # missing value imputation using KNN
        impute_data = self.perform_missing_values_imputation(norm_data)

        # #####################################  write data to csv, to b deleted later#####################
        df = pd.DataFrame(impute_data)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "DataPostImpute.csv", index=False, header=False)
        # data after data impute
        #################################################################################################

        # check for duplicate features. If duplicate then take its average
        mean_data, mean_proteins = self.average_duplicate_measurements(impute_data, new_proteins)

        # #####################################  write data to csv, to b deleted later#####################
        df = pd.DataFrame(mean_data)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "DataPostDupAvg.csv", index=False, header=False)
        # data after duplicate feature average

        df = pd.DataFrame(mean_proteins)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "OmicsPostDupAvg.csv", index=False, header=False)
        # data after duplicate feature average
        #################################################################################################

        # outlier detection - LOF (Linear outlier filter) on dimensionality reduction using PCA. outliers are deleted
        # outlier_data,class_LabelNew = self.outlier_detection(mean_data,class_Labels)
        outlier_data = mean_data
        class_LabelNew = class_Labels
        # #####################################  write data to csv, to b deleted later#####################
        df = pd.DataFrame(outlier_data)
        df.to_csv(path_or_buf=self.pathProcess + os.sep + "DataPostOutlier.csv", index=False, header=False)
        # data after outlier removal
        #################################################################################################

        np.savetxt(self.pathProcess + os.sep + 'Preprocess_Output.txt', [self.output_message], fmt='%s', newline='\n')
        # print output_message to text

        return outlier_data, mean_proteins, class_LabelNew, dataset_initial,  # return initial data for association rule


# ######## feature selection class ###############################################

class FeatureSelection:

    # Constructor, called at the start of this class
    def __init__(self, dataset, proteins, min_values, max_values, class_Labels, evaluate_values, max_trees, population,
                 generations, folds, goal_values, ImbalanceDataSample, dictCountLabels, dictRatioLabels, Ratiovalues,
                 ImbalanceThreshold, dict_chain, mutation_prob=0.2, arith_crossover_prob=0.00,
                 two_point_crossover=0.90, thread_num=2):

        # variables for these parameters
        self.dataset = dataset
        # self.dataTranspose = dataTran
        self.proteins = proteins
        self.class_Labels = class_Labels  # labels or classes as list of list from pre process step
        '''self.class_Labels_binary = copy.deepcopy(class_Labels)
        for k in range(len(class_Labels)):
            for j in range(len(class_Labels[k])):
                if class_Labels[k][j] >= 0.0:
                    self.class_Labels_binary[k][j] = 1
                else:
                    self.class_Labels_binary[k][j] = 0'''

        self.powerSetClass = []  # initialise powerset class labels
        self.evaluate_values = evaluate_values  # count of parameters to be optimised
        self.dict_chain = dict_chain  # dictionary with permutations of classifier chain
        # self.labels = labels # classification labels

        self.min_values = min_values  # min for initial 4 parameters in the individual solution
        self.max_values = max_values  # max for initial 4 parameters in the individual solution

        self.population = int(population)  # population size in GA
        self.generations = int(generations)  # no of generations in GA
        self.twoCross_prob = float(two_point_crossover)  # 2-point cross over probability
        self.arithCross_prob = float(arith_crossover_prob)  # arithmetic cross over probability
        self.mutation_prob = [0.001, float(mutation_prob)]  # mutation probability, min = 0.001 and max = 0.2
        self.gaussMut_varPro = [0.1, 0.5]  # gaussian variance proportion, min=0.1 and max = 0.5
        self.avgSimThr = 0.90  # average similarity between the members of the population and its best member
        self.folds = int(folds)  # number of folds for cross validation
        # self.fold_size = math.floor(float(len(dataset[0]))/folds) # elements to have in a fold for cross validation

        self.output_message = ''  # output message from feature selection
        self.threshold = 0.5  # threshold for feature selected or rejected in GA
        self.score_types = ['neg_mean_squared_error', 'r2', 'neg_median_absolute_error', 'explained_variance']
        self.goal_header = "Feature,NegativeMSE, R2, Negative_MedianAbsoluteError, explained_variance," \
                           "SVs-Trees-Neurons, Distance, RMSE, correlation, Weighted_Sum, Classification_Model"
        # goal significance for goals position 0 thru 8; includes distance. Tested using diff goal_significance in
        # diff. cycles
        # self.goal_significance = [1.0,1.43,1.43,1.43,1.43,1.43,1.43,1.0,1.43] # option 1 1:5 for goal significance
        self.goal_significance = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0]
        # option 1 1:3.5 for goal significance
        self.goal_values = goal_values
        # self.goal_significance = [3.5,1.0,1.0,1.0,1.0,1.0,1.0,3.5,1.0] # option 2 1:1 for goal significance
        # self.goal_significance = [3.5,2.0,2.0,2.0,2.0,2.0,2.0,3.5,1.0] # option 3 1:2 for goal significance
        self.convergeThreshold = 0.001  # convergence threshold for GA
        self.assocRuleThreshold = 0.6

        # initialise variables for synthetic data generation
        self.dictCountLabels = dictCountLabels  # holds count of samples by classes
        self.ImbalanceThreshold = ImbalanceThreshold  # holds ratio guided by class count for imbalance detection
        self.ImbalanceDataSample = ImbalanceDataSample  # flag that says whether or not to boost the imbalanced class
        # data with sampling
        # print ("sampling flag in init is %s"%self.ImbalanceDataSample)
        self.Ratiovalues = Ratiovalues  # holds list of ratios of the classes
        self.dictRatioLabels = dictRatioLabels  # hold ratio of samples by classes
        self.classification_problems = 0  # counts of one vs one for wilcoxon rank sum
        self.max_trees = max_trees  # used for computation of performance measure for complexity of random forest
        self.max_neurons = 512  # max neurons for every layer, performance measure for complexity of neural network
        # model variables for goal values (post pareto front goal tuning) for visualisation
        self.max_eval_per_generation = [0] * self.generations  # best values per goal per generation
        self.average_eval_per_generation = [0] * self.generations  # weighted avg value per generation across all goals
        self.sum_ranked_eval_per_generation = [0] * self.generations  # Sum of weighted average goal per generation
        self.average_ranked_eval_per_generation = [0] * self.generations  # Avg of weighted average goal per generation
        self.max_sol_per_generation = [0] * self.generations  # best sol per generation
        # self.max_sol_per_generation = [[] for _ in range(self.generations)] #initialise list to hold best sol per gen
        self.Pareto_Wheel_Output = ''  # output message from Pareto front and roulette wheel

        self.pathFeature = ''
        self.pathModel = ''
        self.num_threads = 6

    # Accessor functions for parameters
    def setPopulation(self, value):
        self.population = value

    def getPopulation(self):
        return self.population

    def setGenerations(self, value):
        self.generations = value

    def getGenerations(self):
        return self.generations

    def setFolds(self, value):
        self.folds = value

    def getFolds(self):
        return self.folds

    def setGoalSignificanceFeatures(self, value):
        self.goal_significance[0] = float(value)

    def setGoalSignificanceMSE(self, value):
        self.goal_significance[1] = float(value)

    def setGoalSignificanceR2(self, value):
        self.goal_significance[2] = float(value)

    def setGoalSignificanceMAE(self, value):
        self.goal_significance[3] = float(value)

    def setGoalSignificanceExplainedVariance(self, value):
        self.goal_significance[4] = float(value)

    def setGoalSignificanceModelComplexity(self, value):
        # weight for number of SVs or Trees in randomforest
        self.goal_significance[5] = float(value)

    def setGoalSignificanceDistance(self, value):
        self.goal_significance[6] = float(value)

    def setGoalSignificanceRMSE(self, value):
        self.goal_significance[7] = float(value)

    def setGoalSignificanceCorrelation(self, value):
        self.goal_significance[8] = float(value)

    def getGoalSignificances(self):
        return self.goal_significance

    def setGoalSignificancesByUser(self, feature_significance, mse_significance, model_complexity_significance):
        self.setGoalSignificanceFeatures(feature_significance)
        self.setGoalSignificanceMSE(mse_significance)
        self.setGoalSignificanceModelComplexity(model_complexity_significance)

    def setGoalSignificancesByUserList(self, goal_significances):
        self.setGoalSignificanceFeatures(goal_significances[0])
        self.setGoalSignificanceMSE(goal_significances[1])
        self.setGoalSignificanceR2(goal_significances[2])
        self.setGoalSignificanceMAE(goal_significances[3])
        self.setGoalSignificanceExplainedVariance(goal_significances[4])
        self.setGoalSignificanceModelComplexity(goal_significances[5])
        self.setGoalSignificanceRMSE(goal_significances[6])
        self.setGoalSignificanceCorrelation(goal_significances[7])

    def setPaths(self, featurepath, modelpath):
        self.pathFeature = featurepath
        self.pathModel = modelpath

    def getPathFeature(self):
        return self.pathFeature

    def getPathModel(self):
        return self.pathModel

    # This function initializes a population of solutions (float vectors) for optimisation parameters
    # This function is called by biomarker_feature_selection()
    def initialize(self):
        population = self.getPopulation()  # population:(integer) it is the number of individual solutions which are
        # evolved on parallel
        min_values = self.min_values  # min_values: (list) it is a list with the minimum allowed values for the
        # variables which should be optimized
        max_values = self.max_values  # max_values: (list) it is a list with the maximum allowed values for the
        # variables which should be optimized

        # extend min and max values for length of features. Min-max guide length of individuals initialised for GA
        # population
        min_values.extend([0] * len(self.dataset))
        max_values.extend([1] * len(self.dataset))

        # initialise population count (size) as list of list with each individual =  parameters + feature length
        individuals = [[0 for x in range(len(min_values))] for x in range(population)]
        # generate random values for each individual in the population
        for i in range(len(individuals)):  # iterate on population size i.e. count of individuals
            for j in range(0, len(min_values)):  # iterate on size of each individual
                if j == 10 | j == 11:
                    individuals[i][j] = random.randint(min_values[j],
                                                       max_values[j])  # assign values between min and max at
                else:
                    individuals[i][j] = random.uniform(min_values[j],
                                                       max_values[j])  # assign values between min and max at

        # random

        self.output_message += "Population initialised for %s" % len(individuals) + " individuals \n"
        return individuals

    # function to split data by class; used by wilcoxon ranksum for a specific feature index
    def splitData_perFeat_byClass(self, feature_index, lblClassCount, lbl):

        listClassData = []  # initialise list of list to hold data by class for a given feature index
        classIndex = 0
        # split the dataset into class labels for each feature
        for k in lblClassCount.keys():
            listClassData.append([])
            for j in range(len(self.dataset[0])):  # segregate observations per protein i.e. feature
                if int(self.class_Labels[lbl][j]) == int(k):
                    value = self.dataset[feature_index][j]
                    try:
                        listClassData[classIndex].append(float(value))
                    except ValueError:
                        continue
            classIndex = classIndex + 1  # increment the counter for class

        return listClassData  # returns list of list

    def wilcoxon_rank_sum_test(self):
        """
        function to perform wilcoxon rank sum - non-parametric 2-sample t-test for not normal distribution
        null hypothesis is equal means; performed for each feature i.e. protein
        this function is called by biomarker_feature_selection()
        :return:
        """
        pvals = [[] for _ in range(len([1]))]  # list to list to hold features by label such as
        # Diabetes, Depression
        pvalFinal = []
        feature_count = len(self.proteins)
        # run wilconxon rank-sum for each feature and each label such as Diabetes
        for lbl, key in enumerate([1]):  # loop  for labels i.e. single or multi-label
            # print("Ranksum for label {}".format(key))
            lblClassCount = dict(Counter(self.class_Labels[lbl]))
            # print(lblClassCount)  # {0.0:275,1.0:21}
            self.classification_problems = len(lblClassCount)  # count of classes by label
            # loop for features i.e. omics
            for feature in range(feature_count):  # row i.e feature
                pAllVals = []  # reset all pVals for each feature
                listClassData = self.splitData_perFeat_byClass(feature, lblClassCount, lbl)  # split the data by
                # classes for the given feature and label
                # self.classification_problems holds class count. Do one vs All comparisons for each class
                pval = 1
                for i in range(self.classification_problems):
                    for j in range(i + 1, self.classification_problems):
                        if len(listClassData[i]) > 1 and len(listClassData[j]) > 1:  # avoid samples with length 1
                            if statistics.stdev(listClassData[i]) == 0 and statistics.stdev(listClassData[j]) == 0 \
                                    and statistics.mean(listClassData[i]) == statistics.mean(listClassData[j]):
                                pval = 1  # initialise pval to 1
                            else:
                                [z, pval] = st.ranksums(listClassData[i], listClassData[j])
                        else:
                            # print ("single length data")
                            [z, pval] = st.ranksums(listClassData[i], listClassData[j])
                    # pval=0.0001 # as min for pval is 0.001 and feature selection check is pVal < GA pVAl threshold
                    pAllVals.append(pval)

                # take min pval for each feature and append to final pval list
                pvals[lbl].append(min(pAllVals))

        df = pd.DataFrame(pvals)
        df.to_csv(path_or_buf=self.pathFeature + "PValMultiLbl.csv", index=False, header=False)  # pval data

        for i in range(feature_count):  # row i.e feature
            pvalFinal.append(np.min(np.array(pvals)[:, i]))  # take min pval across labels

        df = pd.DataFrame(pvalFinal)
        df.to_csv(path_or_buf=self.pathFeature + "PValFinal.csv", index=False, header=False)  # pval data

        # return pvals
        return pvalFinal

    '''#Ref: SelectKBest: 
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#
    sklearn.feature_selection.SelectKBest
    uses k= features randomized between min and max in GA solutions this function is called by 
    biomarker_feature_selection()'''

    def kbest_features(self, dataTran, kFeature, kBestFeatFinal, lbl):
        # X: numpy array of shape [n_samples, n_features]
        # Attribute scores_ : array-like, shape=(n_features,) Scores of features.
        # pvalues_ : array-like, shape=(n_features,) p-values of feature scores, None if score_func returned only
        # scores.

        kBestFeat = []
        # kbest for each label such as Diabetes, depression
        for l, key in enumerate(lbl):  # loop  for labels i.e. single or multi-label
            # default scoring function is ANOVA F-value between label/feature for classification tasks.
            # F-test captures only linear dependency, but mutual information can capture any kind of dependency between
            # variables
            kBestFeat.append([0 for _ in range(len(self.proteins))])
            kbest = SelectKBest(score_func=mutual_info_regression, k=kFeature)
            kbest.fit(dataTran, self.class_Labels[l])  # labels is list of list
            # print ("kest support")
            # print (kbest.get_support())
            kBestFeat[l] = kbest.get_support()  # get_support() gives array - with index that selects the retained
            # features
            # from a feature vector.

        for i in range(len(self.proteins)):  # row i.e feature
            kBestFeatFinal[i] = (np.max(np.array(kBestFeat)[:, i]))  # take max across labels

        # If indices is False (default), this is a boolean array of shape [# input features]
        # return kbest.get_support()
        # return kBestFeat

        return kBestFeatFinal

    '''#function for joint mutual information maximisation JMIM or mRMR i.e. max. relevance and min redundancy
    # Ref: https://github.com/danielhomola/mifs
    # this function is called by biomarker_feature_selFitnessection()'''

    # def MutualInfo_features(self,dataTran,MIneighbour,kFeature,MImethod):
    def MutualInfo_features(self, dataTran, MIneighbour, MImethod):
        # X: numpy array of shape [n_samples, n_features]
        # returns mi : ndarray, shape (n_features,) Estimated mutual information between each feature and the target.

        # define MIFS feature selection method;
        # n_features = default i.e. auto : to be determined automatically based on the amount of MI the previously
        # selected features share with y
        # categorical = default i.e. True. If True, y is assumed to be a categorical class label else continuous
        # n_jobs = -1 for parallelism
        MIBestFeat = []  # list to hold features by label such as Diabetes, Depression
        MIBestFeatFinal = []
        # print (MULTI_LABELS)
        # kbest for each label such as Diabetes
        for lbl, key in enumerate([1]):  # loop  for labels i.e. single or multi-label
            MIbest = mifs.MutualInformationFeatureSelector(method=MImethod, k=MIneighbour, n_features='auto',
                                                           n_jobs=self.num_threads)
            # find all relevant features; change labels to int64 for categorical classes
            MIbest.fit(dataTran, np.array(self.class_Labels[lbl], dtype=np.int64))  # labels is list of list
            MIBestFeat.append(MIbest.support_)

        for i in range(len(self.proteins)):  # row i.e feature
            MIBestFeatFinal.append(np.max(np.array(MIBestFeat)[:, i]))  # take max across labels

        # return MIbest.support_
        # return MIBestFeat
        return MIBestFeatFinal

    def ROC_measures(self, y_actual, y_predict):  # function to compute tp, tn, fp, fn for ROC
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(y_predict)):
            if y_actual[i] == y_predict[i] == 1:
                TP += 1
            if y_predict[i] == 1 and y_actual[i] != y_predict[i]:
                FP += 1
            if y_actual[i] == y_predict[i] == 0:
                TN += 1
            if y_predict[i] == 0 and y_actual[i] != y_predict[i]:
                FN += 1
        return TP, FP, TN, FN

    @staticmethod
    def neural_model(learning_rate, input, layers, neurons, opt, drop):
        neural = keras.Sequential()

        for i in range(layers):
            if i == 0:  # we need input shape for first layer
                neural.add(Conv1D(filters=neurons, kernel_size=1, strides=1, activation='relu', input_shape=(input, 1)))
            else:
                neural.add(Conv1D(filters=neurons, kernel_size=1, strides=1, activation='relu'))

            neural.add(Dropout(drop))
            neural.add(MaxPooling1D(2, padding='same'))

        neural.add(Flatten())
        neural.add(Dense(128, activation='relu'))
        neural.add(Dense(1))
        if opt < 1:
            optimizer = Adam(learning_rate=learning_rate)
        elif 1 <= opt < 2:
            optimizer = RMSprop(learning_rate=learning_rate)
        elif 2 <= opt < 3:
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = Nadam(learning_rate=learning_rate)
        neural.compile(loss='mae', optimizer=optimizer, metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
        return neural

    @staticmethod
    def get_total_number_of_neurons(model, include_output_layer):
        """
        Args:
            model: Keras model
            include_output_layer: A boolean parameter. Whether or not to include output layer's neurons into the
            calculation

        Returns: number of neurons in the given model
        """
        num_layers = len(model.layers)
        total_num_of_neurons = 0
        for layer_index in range(num_layers):
            layer = model.layers[layer_index]
            # since we multiply layer output dimensions, initial value is set to 1.
            num_neurons_in_layer = 1
            for i in range(1, len(layer.output.shape)):
                try:
                    # when it is a valid layer to count neurons, an output dimension of the layer can
                    # be convertible to int.
                    num_neurons_in_layer *= int(layer.output.shape[i])
                except Exception:
                    # if the output dimension of layer cannot be convertible to int,
                    # just pass that layer since it is not a valid layer to count neurons
                    pass
            # if num_neurons_in_layer is not still 1, it means we have a valid layer to count neurons
            if not num_neurons_in_layer == 1:
                # when it is an output layer
                if layer_index == (num_layers - 1):
                    if include_output_layer:
                        total_num_of_neurons += num_neurons_in_layer
                else:  # when it is not an output layer
                    total_num_of_neurons += num_neurons_in_layer
        return total_num_of_neurons

    '''# function to calculate remaining goal values (as per the fitness function) for each of the individual solutions
     in a given population
    # goal_values initialized to zero and calculated in this function
    # This function is called by evaluate_indivdiuals() for each individual solution if the problem is multi-label.
     final_selected is features post intersection with GA.
    # classifier_type - is the value of the 1st position of the individual solution that decides the classifier type'''

    def train_Classifier_MultiLabel(self, dataTran, final_selected, classifier_type, C_val, Gamma_val, Classchain,
                                    TreeCount, ftwo_scorer, finalFlag, multilabel, lrate, epochs, batchsize, layers,
                                    neurons, optimizer, dropout):
        inputs = []  # initialise X of the model
        Colindices = []  # feature indices selected
        outputs_copy = deepcopy(self.class_Labels)  # assign to a variable so that synthetic data labels can be appended
        # Create classification model dataset aligned with features selected
        # X i.e. input to be of the form shape [n_samples, n_features]
        # dataTRan is np.ndarray with shape as required by X
        for feature in range(len(final_selected)):
            if final_selected[feature] == 1:  # append only those feature that are 1 i.e. selected
                Colindices.append(feature)  # get the indices of features selected

        self.output_message += "Feature indices in final selection : %s" % Colindices + "\n"
        # append all rows i.e. observations for the feature selected

        inputs = dataTran[:, Colindices]  # type = numpy array
        # print("Input dataset created for selected features with shape : ")
        # print(inputs.shape)

        # Create classification model dataset aligned with features selected
        outputs = []  # initialise empty list
        if int(Classchain) == 0:  # retain original sequence
            # print("Original classifier chain selected")
            outputs = copy.deepcopy(outputs_copy)  # no shuffling req.
        else:
            original_seq = copy.deepcopy(multilabel)
            # get the new sequence from dictionary
            try:
                new_seq = self.dict_chain[int(Classchain)]
                # print("New classifier chain selected")
            except KeyError:
                # print("%s key not found in dictionary" % int(Classchain))
                outputs = copy.deepcopy(outputs_copy)  # don't do shuffling
                # print("Original classifier chain selected")
                new_seq = []

            # re-sequence the outputs
            for i in range(len(new_seq)):
                for j in range(len(original_seq)):
                    if original_seq[j] == new_seq[i]:
                        # add position i to outputs
                        outputs.append([])
                        outputs[i].extend(outputs_copy[j])

        # ######################################################################
        # ##################### Choose the classifier ##########################
        # print("training multi-label classifier")

        SV_Tree = 0.0001  # assign insignificantly low value so that goal value is not zero
        Dist_Metric = 0.0001
        if classifier_type < 1.0:  # linear svm
            # clf = svm.SVC(C=C_val, kernel='linear', random_state=SEED,probability=True) # probability to compute ROC
            clf = svm.SVR(C=C_val, kernel='linear')  # probability to compute ROC
        elif 1.0 <= classifier_type < 2.0:  # RBF svm
            # clf = svm.SVC(C=C_val, gamma= Gamma_val, kernel='rbf', random_state=SEED,probability=True)
            # probability to compute ROC
            clf = svm.SVR(C=C_val, gamma=Gamma_val, kernel='rbf')
        else:  # random forest, n_jobs=-1 for parallelism
            # clf = RandomForestClassifier(n_estimators = int(TreeCount), random_state=SEED,n_jobs=-1)
            clf = RandomForestRegressor(n_estimators=int(TreeCount))

            SV_Tree = float((self.max_trees - int(TreeCount)) / self.max_trees)
            # don't round so that insignificantly low values are kept

        lbl = len(multilabel)  # get the count of labels in multilabel
        models_list = []  # list to hold the various model fitting

        # flag to check if this is for final model save or evaluation of solutions during GA

        if finalFlag:  # if final model being trained then no CV required
            for i in range(lbl):
                clf_copy = clone(clf)  # take deepcopy of model
                if i == 0:  # 1st label uses X without additional features i.e. labels
                    clf_copy.fit(inputs, np.array(outputs).T[:, 0])  # read the 1st column i.e. 1st label for fitting
                # y_pred = (clf_copy.predict(inputs)).reshape(-1, 1)
                else:
                    # Stack arrays in sequence horizontally (column wise).
                    # select all labels but for the ith label which needs to be predicted
                    input_stacked = np.hstack((inputs, np.array(outputs).T[:, :i]))
                    clf_copy.fit(input_stacked, np.array(outputs).T[:, i])  # for output, select only the ith label
                    # i.e. single label

                models_list.append(clf_copy)  # append the models trained on diff data
            # print("final model fitting done")
            return models_list, inputs, outputs

        mskf = MultilabelStratifiedKFold(n_splits=self.getFolds(), shuffle=False)  # stratified to preserve the
        # percentage of samples for each
        # class.
        sv_list_fold = []  # initialise list for computing average of sv score at folds level
        sv_list_model = []  # initialise list for computing average of sv score at classifier chain level
        cv_scores = []  # initialise list of performance scores averaged over k-folds
        reg_scores = []
        # initialise performance metrics to compute at model level and then aggregate at fold level
        f1_score_model = []
        f2_score_model = []
        precision_model = []
        recall_model = []
        roc_model = []
        accuracy_model = []
        mse_model = []
        correlation_model = []
        f1_score_fold = []
        f2_score_fold = []
        precision_fold = []
        recall_fold = []
        roc_fold = []
        distance_fold = []  # this measures accuracy for each label position
        accuracy_fold = []  # this measures accuracy as all labels predicted correct
        mse_fold = []
        correlation_fold = []

        # use balanced data so that its 1:1 for all labels, else ratio of 1st label will be taken for others
        for train_index, test_index in mskf.split(inputs, np.array(outputs).T):
            # takes 20% as test data, use 1st label to get indexes
            models_list = []  # list to hold the various model fitting
            X_train, X_test = inputs[train_index, :], inputs[test_index, :]  # inputs is ndarray so use np.ndarray
            # slicing
            y_train, y_test = np.array(outputs)[:, train_index], np.array(outputs)[:, test_index]  # get all columns

            # fit on train data using classifier chain

            for i in range(lbl):
                clf_copy = clone(clf)  # take deepcopy of model
                if i == 0:  # 1st label uses X without additional features i.e. labels
                    # y_train, y_test = np.array(outputs[0])[train_index], np.array(outputs[0])[test_index]
                    clf_copy.fit(X_train, y_train.T[:, 0])  # read the 1st column i.e. 1st label for fitting
                # y_pred = (clf_copy.predict(X_test)).reshape(-1, 1)
                else:
                    # Stack arrays in sequence horizontally (column wise).
                    # select all labels but for the ith label which needs to be predicted
                    input_stacked = np.hstack((X_train, y_train.T[:, :i]))
                    # print("Shape after stacking of Xtrain and ytrain : ")
                    # print (input_stacked.shape)

                    clf_copy.fit(input_stacked, y_train.T[:, i])
                    # for output, select only the ith label i.e. single label

                models_list.append(clf_copy)  # append the models trained on diff data
                train_size = (len(X_train))
                if classifier_type < 2.0:  # svm classifier
                    sv_size = len(clf_copy.support_)
                    # print("SVs used %s" % sv_size)
                    if train_size > 0:
                        sv_list_model.append(
                            float(len(list(y_train.T[:, 0])) - sv_size) / len(list(y_train.T[:, 0])))  # don't round

            if classifier_type < 2.0:
                sv_list_fold.append(np.mean(np.array(sv_list_model)))
            # sv_list_fold.append(0.0001 if np.isnan(np.array(sv_list_model)) else np.nanmean(np.array(sv_list_model)))

            # print("multi label fitting done")

            if len(models_list) == 0:  # test for parallel processing session hanging without error
                with open('ERRORModelList.txt', 'w') as fmodel:  # redirect all print statements to file output
                    print("empty model list found in multi label", file=fmodel)

                f2_score_model.append(0.0001)
                f1_score_model.append(0.0001)
                precision_model.append(0.0001)
                recall_model.append(0.0001)
                roc_model.append(0.0001)
                accuracy_model.append(0.0001)
                correlation_model.append(0.0001)
                mse_model.append(0.0001)

            for i, model in enumerate(models_list):  # predict on list of models for a give train/ test data
                if i == 0:
                    y_pred_reg = (model.predict(X_test)).reshape(-1, 1)  # reshape to 1 column and rows as relevant

                    # del y_pred_temp

                    y_pred = copy.deepcopy(y_pred_reg)
                    # y_pred=list()
                    for f in range(len(y_pred_reg)):
                        if y_pred_reg[f] >= 0:
                            y_pred[f] = 1
                        else:
                            y_pred[f] = 0
                    y_test_reg = copy.deepcopy(y_test)
                    y_test_new = copy.deepcopy(y_test)
                    for f1 in range(len(y_test_reg)):
                        for f2 in range(len(y_test_reg[f1])):
                            if y_test_reg[f1][f2] >= 0:
                                y_test_new[f1][f2] = 1
                            else:
                                y_test_new[f1][f2] = 0

                    # print(y_pred_reg)
                    # print(list(y_test_reg.T[:, 0]))

                    # with open('ypred.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_pred_reg)
                    # with open('ytest.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_test_reg.T[:, 0])

                    # print ("y shape")
                    # print (y_pred.shape) # (60,1)
                    # compute performance score i.e. ['accuracy','f1_weighted','precision_weighted', 'recall_weighted',
                    # 'roc_auc']
                    # print(y_pred)
                    # print(y_test.T[:, 0])

                    # print("Unpredicted labels:")
                    # print(set(list(y_test.T[:, 0])) - set(list(y_pred)))
                    f2_score_model.append(fbeta_score(y_test_new.T[:, 0], y_pred, average='weighted', beta=2.0))
                    f1_score_model.append(fbeta_score(y_test_new.T[:, 0], y_pred, average='weighted', beta=1.0))
                    precision_model.append(precision_score(y_test_new.T[:, 0], y_pred, average='weighted'))
                    recall_model.append(recall_score(y_test_new.T[:, 0], y_pred, average='weighted'))
                    # AUC has 2 parts: a triangle and a trapezium. Triangle area = TPR*FPR/2,
                    # the trapezium (1-FPR)*(1+TPR)/2 = 1/2 - FPR/2 + TPR/2 - TPR*FPR/2.
                    # Hence, sum of the two = 1/2 - FPR/2 + TPR/2.
                    # FPR = fp / (fp + tn); TPR = tp / (tp + fn)
                    # TP, FP, TN, FN = self.ROC_measures(y_test.T[:, 0], model.predict(X_test))
                    # FPR = float(FP/(FP+TN))
                    # TPR = float(TP/(TP+FN))
                    # roc_model.append(0.5 -FPR/2.0 + TPR/2.0)
                    # y_score = model.predict_proba(X_test) # 60,2 for 2 classes

                    # with open('ypred_reg.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_pred_reg)
                    # with open('ytest_reg.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_test_reg.T[:, 0])
                    try:
                        val = roc_auc_score(y_test_new.T[:, 0], y_pred_reg, average='weighted')
                    except:
                        with open('ROC_exception.txt', 'w') as myfile:
                            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                            wr.writerow(str(i))
                        val = 0

                    roc_model.append(val)

                    accuracy_model.append(accuracy_score(y_test_new.T[:, 0], y_pred))
                    # to get distance or accuracy based on hamming loss
                    mse_model.append(1.0 / (1 + mean_squared_error(y_test_reg.T[:, 0], y_pred_reg)))

                    [r, p] = st.spearmanr(y_test_reg.T[:, 0], y_pred_reg)
                    if math.isnan(r) or r < 0:
                        correlation_model.append(0)
                    else:
                        correlation_model.append(r)
                else:
                    input_stacked = np.hstack((X_test, y_pred))  # add the prior y or Ys predicted to X
                    # print("Shapes after stacking of Xtest : ")
                    # print (input_stacked.shape)

                    new_y_reg = model.predict(input_stacked)  # predict y using the model trained for X and Y

                    new_y = copy.deepcopy(new_y_reg)

                    for f in range(len(new_y_reg)):
                        if new_y_reg[f] >= 0:
                            new_y[f] = 1
                        else:
                            new_y[f] = 0
                    y_test_reg = copy.deepcopy(y_test)
                    y_test_new = copy.deepcopy(y_test)
                    for f1 in range(len(y_test_reg)):

                        for f2 in range(len(y_test_reg[f1])):
                            if y_test_reg[f1][f2] >= 0:
                                y_test_new[f1][f2] = 1
                            else:
                                y_test_new[f1][f2] = 0

                    # print(new_y_reg)
                    # print(list(y_test_reg.T[:, i]))

                    # print(set(y_test.T[:, 0]) - set(new_y))
                    # compute performance score i.e. ['accuracy','f1_weighted','precision_weighted', 'recall_weighted',
                    # 'roc_auc']
                    f2_score_model.append(fbeta_score(y_test_new.T[:, i], new_y, average='weighted', beta=2.0))
                    f1_score_model.append(fbeta_score(y_test_new.T[:, i], new_y, average='weighted', beta=1.0))
                    precision_model.append(precision_score(y_test_new.T[:, i], new_y, average='weighted'))
                    recall_model.append(recall_score(y_test_new.T[:, i], new_y, average='weighted'))
                    # AUC has 2 parts: a triangle and a trapezium. Triangle area = TPR*FPR/2,
                    # the trapezium (1-FPR)*(1+TPR)/2 = 1/2 - FPR/2 + TPR/2 - TPR*FPR/2.
                    # Hence, sum of the two = 1/2 - FPR/2 + TPR/2.
                    # FPR = fp / (fp + tn); TPR = tp / (tp + fn)
                    # TP, FP, TN, FN = self.ROC_measures(y_test.T[:, i], new_y)
                    # FPR = float(FP/(FP+TN))
                    # TPR = float(TP/(TP+FN))
                    # roc_model.append(0.5 -FPR/2.0 + TPR/2.0)
                    # y_score = model.predict_proba(input_stacked)
                    # print(new_y)
                    # print(list(y_test.T[:, i]))

                    try:
                        val = roc_auc_score(y_test_new.T[:, i], new_y_reg, average='weighted')
                    except:
                        val = 0
                    roc_model.append(val)

                    accuracy_model.append(
                        accuracy_score(y_test_new.T[:, i], new_y))  # to get distance or accuracy based on hamming loss
                    mse_model.append(1.0 / (1 + mean_squared_error(y_test_reg.T[:, i], new_y_reg)))
                    [r, p] = st.spearmanr(y_test_reg.T[:, i], y_pred_reg)
                    if math.isnan(r) or r < 0:
                        correlation_model.append(0)
                    else:
                        correlation_model.append(r)

                    y_pred = np.hstack((y_pred, new_y.reshape(-1, 1)))  # append the Ys predicted
            # print ("y predict shape after stacking ")
            # print (y_pred.shape)

            # take average of metrics at model level
            f2_score_fold.append(np.mean(np.array(f2_score_model)))
            f1_score_fold.append(np.mean(np.array(f1_score_model)))
            precision_fold.append(np.mean(np.array(precision_model)))
            recall_fold.append(np.mean(np.array(recall_model)))

            roc_fold.append(np.mean(np.array(roc_model)))

            mse_fold.append(np.mean(np.array(mse_model)))
            correlation_fold.append(np.mean(np.array(correlation_model)))

            mismatch_count = zero_one_loss(y_test_new.T, y_pred, normalize=False)
            # measures count of mismatch based on zero-one-loss i.e. entire set of labels must be correct else loss for
            # that sample = 1

            accuracy_fold.append(np.mean(np.array(accuracy_model)))
            distance_fold.append(1.0 - float(mismatch_count / len(X_test)))
            # measure accuracy for each label, hence distance based on hamming loss
        # self.output_message += "predicted Y : %s"%y_pred + "\n"
        # self.output_message += "original Y : %s"%y_test + "\n"

        # take average of metrics at fold level and append in sequence
        # ['accuracy','f1_weighted','precision_weighted', 'recall_weighted','roc_auc']
        cv_scores.append(np.mean(np.array(accuracy_fold)))
        cv_scores.append(np.mean(np.array(f1_score_fold)))
        cv_scores.append(np.mean(np.array(precision_fold)))
        cv_scores.append(np.mean(np.array(recall_fold)))
        cv_scores.append(np.mean(np.array(roc_fold)))
        reg_scores.append(np.mean(np.array(mse_fold)))
        reg_scores.append(np.mean(np.array(correlation_fold)))
        # print("reg_scores=")
        # print(reg_scores)
        f2_score = np.mean(np.array(f2_score_fold))
        if classifier_type < 2.0:  # svm classifier
            SV_Tree = 0.0001 if np.isnan(np.nanmean(np.array(sv_list_fold))) else np.nanmean(np.array(sv_list_fold))
        # else:
        # SV_Tree = 0.0001
        # print("SV_tree")
        # print(SV_Tree)
        Dist_Metric = np.mean(np.array(distance_fold))
        # print(Dist_Metric)
        if SV_Tree < 0 or SV_Tree > 1:
            SV_Tree = 0
        return cv_scores, SV_Tree, f2_score, Dist_Metric, reg_scores

    def train_Classifier_MultiLabel2(self, dataTran, final_selected, classifier_type, C_val, Gamma_val, Classchain,
                                     TreeCount, ftwo_scorer, finalFlag, multilabel, num_lab):
        inputs = []  # initialise X of the model
        Colindices = []  # feature indices selected
        outputs_copy = deepcopy(self.class_Labels)  # assign to a variable so that synthetic data labels can be appended
        # Create classification model dataset aligned with features selected
        # X i.e. input to be of the form shape [n_samples, n_features]
        # dataTRan is np.ndarray with shape as required by X
        for feature in range(len(final_selected)):
            if final_selected[feature] == 1:  # append only those feature that are 1 i.e. selected
                Colindices.append(feature)  # get the indices of features selected

        self.output_message += "Feature indices in final selection : %s" % Colindices + "\n"
        # append all rows i.e. observations for the feature selected
        inputs = dataTran[:, Colindices]  # type = numpy array
        # print("Input dataset created for selected features with shape : ")
        # print(inputs.shape)

        # Create classification model dataset aligned with features selected
        outputs = []  # initialize empty list
        if int(Classchain) == 0:  # retain original sequence
            # print("Original classifier chain selected")
            outputs = copy.deepcopy(outputs_copy)  # no shuffling req.
        else:
            original_seq = copy.deepcopy(multilabel)
            # get the new sequence from dictionary
            try:
                new_seq = self.dict_chain[int(Classchain)]
                # print("New classifier chain selected")
            except KeyError:
                # print("%s key not found in dictionary" % int(Classchain))
                outputs = copy.deepcopy(outputs_copy)  # don't do shuffling
                # print("Original classifier chain selected")

            # resequence the outputs
            for i in range(len(new_seq)):
                for j in range(len(original_seq)):
                    if original_seq[j] == new_seq[i]:
                        # add position i to outputs
                        outputs.append([])
                        outputs[i].extend(outputs_copy[j])

        # ######################################################################
        # ##################### Choose the classifier ##########################
        # print("training multi-label classifier")

        SV_Tree = 0.0001  # assign insignificantly low value so that goal value is not zero
        Dist_Metric = 0.0001
        if classifier_type < 1.0:  # linear svm
            # clf = svm.SVC(C=C_val, kernel='linear', random_state=SEED,probability=True) # probability to compute ROC
            clf = svm.SVR(C=C_val, kernel='linear')  # probability to compute ROC
        elif 1.0 <= classifier_type < 2.0:  # RBF svm
            # clf = svm.SVC(C=C_val, gamma= Gamma_val, kernel='rbf', random_state=SEED,probability=True)
            # probability to compute ROC
            clf = svm.SVR(C=C_val, gamma=Gamma_val, kernel='rbf')
        else:  # random forest, n_jobs=-1 for parallelsim
            # clf = RandomForestClassifier(n_estimators = int(TreeCount), random_state=SEED,n_jobs=-1)
            clf = RandomForestRegressor(n_estimators=int(TreeCount))

            SV_Tree = float((self.max_trees - int(TreeCount)) / self.max_trees)
            # don't round so that insignificantly low values are kept

        lbl = len(multilabel)  # get the count of labels in multilabel
        models_list = []  # list to hold the various model fitting

        # flag to check if this is for final model save or evaluation of solutions during GA

        if finalFlag:  # if final model being trained then no CV required
            for i in range(lbl):
                clf_copy = clone(clf)  # take deepcopy of model
                if i == 0:  # 1st label uses X without additional features i.e. labels
                    clf_copy.fit(inputs, np.array(outputs).T[:, 0])  # read the 1st column i.e. 1st label for fitting
                # y_pred = (clf_copy.predict(inputs)).reshape(-1, 1)
                else:
                    # Stack arrays in sequence horizontally (column wise).
                    # select all labels but for the ith label which needs to be predicted
                    input_stacked = np.hstack((inputs, np.array(outputs).T[:, :i]))
                    clf_copy.fit(input_stacked,
                                 np.array(outputs).T[:, i])  # for output, select only the ith label i.e. single label

                models_list.append(clf_copy)  # append the models trained on diff data
            # print("final model fitting done")
            return models_list, inputs, outputs

        mskf = MultilabelStratifiedKFold(n_splits=self.getFolds(), shuffle=False)  # stratified to
        # preserve the percentage of samples for each class.
        sv_list_fold = []  # initialise list for computing average of sv score at folds level
        sv_list_model = []  # initialise list for computing average of sv score at classifier chain level
        cv_scores = []  # initialise list of performance scores averaged over k-folds
        reg_scores = []
        # initialise performance metrics to compute at model level and then agreggate at fold level
        f1_score_model = []
        f2_score_model = []
        precision_model = []
        recall_model = []
        roc_model = []
        accuracy_model = []
        mse_model = []
        correlation_model = []
        f1_score_fold = []
        f2_score_fold = []
        precision_fold = []
        recall_fold = []
        roc_fold = []
        distance_fold = []  # this measures accuracy for each label position
        accuracy_fold = []  # this measures accuracy as all labels predicted correct
        mse_fold = []
        correlation_fold = []

        # use balanced data so that its 1:1 for all labels, else ratio of 1st label will be taken for others
        for train_index, test_index in mskf.split(inputs, np.array(
                outputs).T):  # takes 20% as test data, use 1st label to get indexes
            models_list = []  # list to hold the various model fitting
            X_train, X_test = inputs[train_index, :], inputs[test_index, :]
            # inputs is ndarray so use np.ndarray slicing
            y_train, y_test = np.array(outputs)[:, train_index], np.array(outputs)[:, test_index]  # get all columns

            # fit on train data using classifier chain
            # print("lbl")
            # print(lbl)
            for i in range(lbl):
                clf_copy = clone(clf)  # take deepcoppy of model
                if i == 0:  # 1st label uses X without additional features i.e. labels
                    # y_train, y_test = np.array(outputs[0])[train_index], np.array(outputs[0])[test_index]
                    clf_copy.fit(X_train, y_train.T[:, 0])  # read the 1st column i.e. 1st label for fitting
                # y_pred = (clf_copy.predict(X_test)).reshape(-1, 1)
                else:
                    # Stack arrays in sequence horizontally (column wise).
                    # select all labels but for the ith label which needs to be predicted
                    input_stacked = np.hstack((X_train, y_train.T[:, :i]))
                    # print("Shape after stacking of Xtrain and ytrain : ")
                    # print (input_stacked.shape)

                    clf_copy.fit(input_stacked,
                                 y_train.T[:, i])  # for output, select only the ith label i.e. single label

                models_list.append(clf_copy)  # append the models trained on diff data
                train_size = (len(X_train))
                if classifier_type < 2.0:  # svm classifier
                    sv_size = len(clf_copy.support_)
                    # print("SVs used %s" % sv_size)
                    if train_size > 0:
                        sv_list_model.append(
                            float(len(list(y_train.T[:, 0])) - sv_size) / len(list(y_train.T[:, 0])))  # don't round

            if classifier_type < 2.0:
                sv_list_fold.append(np.mean(np.array(sv_list_model)))
            # sv_list_fold.append(0.0001 if np.isnan(np.array(sv_list_model)) else np.nanmean(np.array(sv_list_model)))

            # print("multi label fitting done")

            if len(models_list) == 0:  # test for parallel processing session hanging without error
                with open('ERRORModelList.txt', 'w') as fmodel:  # redirect all print statements to file output
                    print("empty model list found in multi label", file=fmodel)

                f2_score_model.append(0.0001)
                f1_score_model.append(0.0001)
                precision_model.append(0.0001)
                recall_model.append(0.0001)
                roc_model.append(0.0001)
                accuracy_model.append(0.0001)
                correlation_model.append(0.0001)
                mse_model.append(0.0001)

            for i, model in enumerate(models_list):  # predict on list of models for a give train/ test data

                if i == 0:
                    y_pred_reg = (model.predict(X_test)).reshape(-1, 1)  # reshape to 1 column and rows as relevant

                    # del y_pred_temp

                    y_pred = copy.deepcopy(y_pred_reg)
                    # y_pred=list()
                    for f in range(len(y_pred_reg)):
                        if y_pred_reg[f] >= 0:
                            y_pred[f] = 1
                        else:
                            y_pred[f] = 0

                    y_test_reg = copy.deepcopy(y_test)
                    y_test2 = copy.deepcopy(y_test)

                    for f1 in range(len(y_test_reg)):
                        for f2 in range(len(y_test_reg[f1])):
                            if y_test_reg[f1][f2] >= 0:
                                y_test2[f1][f2] = 1
                            else:
                                y_test2[f1][f2] = 0

                    # print(y_pred_reg)
                    # print(list(y_test_reg.T[:, 0]))

                    # with open('ypred.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_pred_reg)
                    # with open('ytest.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_test_reg.T[:, 0])

                    # print ("y shape")
                    # print (y_pred.shape) # (60,1)
                    # compute performance score i.e. ['accuracy','f1_weighted','precision_weighted', 'recall_weighted',
                    # 'roc_auc']
                    # print(y_pred)
                    # print(y_test.T[:, 0])

                    # print("Unpredicted labels:")
                    # print(set(list(y_test.T[:, 0])) - set(list(y_pred)))
                    if num_lab == 0:
                        f2_score_model.append(fbeta_score(y_test2.T[:, 0], y_pred, average='weighted', beta=2.0))
                        f1_score_model.append(fbeta_score(y_test2.T[:, 0], y_pred, average='weighted', beta=1.0))
                        precision_model.append(precision_score(y_test2.T[:, 0], y_pred, average='weighted'))
                        recall_model.append(recall_score(y_test2.T[:, 0], y_pred, average='weighted'))
                    # AUC has 2 parts: a triangle and a trapezium. Triangle area = TPR*FPR/2,
                    # the trapezium (1-FPR)*(1+TPR)/2 = 1/2 - FPR/2 + TPR/2 - TPR*FPR/2.
                    # Hence, sum of the two = 1/2 - FPR/2 + TPR/2.
                    # FPR = fp / (fp + tn); TPR = tp / (tp + fn)
                    # TP, FP, TN, FN = self.ROC_measures(y_test.T[:, 0], model.predict(X_test))
                    # FPR = float(FP/(FP+TN))
                    # TPR = float(TP/(TP+FN))
                    # roc_model.append(0.5 -FPR/2.0 + TPR/2.0)
                    # y_score = model.predict_proba(X_test) # 60,2 for 2 classes

                    # with open('ypred_reg.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_pred_reg)
                    # with open('ytest_reg.txt', 'w') as myfile:
                    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    # wr.writerow(y_test_reg.T[:, 0])
                    try:
                        val = roc_auc_score(y_test2.T[:, 0], y_pred_reg, average='weighted')
                    except:
                        with open('ROC_exception.txt', 'w') as myfile:
                            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                            wr.writerow(str(i))
                        val = 0

                    if num_lab == 0:
                        roc_model.append(val)

                    if num_lab == 0:
                        accuracy_model.append(accuracy_score(y_test2.T[:, 0], y_pred))
                        # to get distance or accuracy based on hamming loss
                        mse_model.append(1.0 / (1 + mean_squared_error(y_test_reg.T[:, 0], y_pred_reg)))

                    [r, p] = st.spearmanr(y_test_reg.T[:, 0], y_pred_reg)
                    if num_lab == 0:
                        if math.isnan(r) or r < 0:
                            correlation_model.append(0)
                        else:
                            correlation_model.append(r)
                else:
                    input_stacked = np.hstack((X_test, y_pred))  # add the prior y or Ys predicted to X
                    # print("Shapes after stacking of Xtest : ")
                    # print (input_stacked.shape)
                    new_y_reg = model.predict(input_stacked)  # predict y using the model trained for X and Y

                    # print ("y shape predicted")
                    # print (new_y.shape)
                    new_y = copy.deepcopy(new_y_reg)
                    # new_y=list()

                    for f in range(len(new_y_reg)):
                        if new_y_reg[f] >= 0:
                            new_y[f] = 1
                        else:
                            new_y[f] = 0
                    y_test_reg = copy.deepcopy(y_test)

                    y_test2 = copy.deepcopy(y_test)
                    for f1 in range(len(y_test_reg)):

                        for f2 in range(len(y_test_reg[f1])):
                            if y_test_reg[f1][f2] >= 0:
                                y_test2[f1][f2] = 1
                            else:
                                y_test2[f1][f2] = 0

                    # print(new_y_reg)
                    # print(list(y_test_reg.T[:, i]))

                    # print(set(y_test.T[:, 0]) - set(new_y))
                    # compute performance score i.e. ['accuracy','f1_weighted','precision_weighted', 'recall_weighted',
                    # 'roc_auc']
                    if i == num_lab:
                        f2_score_model.append(fbeta_score(y_test2.T[:, i], new_y, average='weighted', beta=2.0))
                        f1_score_model.append(fbeta_score(y_test2.T[:, i], new_y, average='weighted', beta=1.0))
                        precision_model.append(precision_score(y_test2.T[:, i], new_y, average='weighted'))
                        recall_model.append(recall_score(y_test2.T[:, i], new_y, average='weighted'))
                    # AUC has 2 parts: a triangle and a trapezium. Triangle area = TPR*FPR/2,
                    # the trapezium (1-FPR)*(1+TPR)/2 = 1/2 - FPR/2 + TPR/2 - TPR*FPR/2.
                    # Hence, sum of the two = 1/2 - FPR/2 + TPR/2.
                    # FPR = fp / (fp + tn); TPR = tp / (tp + fn)
                    # TP, FP, TN, FN = self.ROC_measures(y_test.T[:, i], new_y)
                    # FPR = float(FP/(FP+TN))
                    # TPR = float(TP/(TP+FN))
                    # roc_model.append(0.5 -FPR/2.0 + TPR/2.0)
                    # y_score = model.predict_proba(input_stacked)
                    # print(new_y)
                    # print(list(y_test.T[:, i]))

                    try:
                        val = roc_auc_score(y_test2.T[:, i], new_y_reg, average='weighted')
                    except:
                        val = 0
                    if i == num_lab:
                        roc_model.append(val)

                    if i == num_lab:
                        accuracy_model.append(
                            accuracy_score(y_test2.T[:, i], new_y))  # to get distance or accuracy based on hamming loss

                    if i == num_lab:
                        mse_model.append(1.0 / (1 + mean_squared_error(y_test_reg.T[:, i], new_y_reg)))

                    [r, p] = st.spearmanr(y_test_reg.T[:, i], y_pred_reg)
                    if i == num_lab:
                        if math.isnan(r) or r < 0:
                            correlation_model.append(0)
                        else:
                            correlation_model.append(r)

                    y_pred = np.hstack((y_pred, new_y.reshape(-1, 1)))  # append the Ys predicted
            # print ("y predict shape after stacking ")
            # print (y_pred.shape)

            # take average of metrics at model level
            f2_score_fold.append(np.mean(np.array(f2_score_model)))
            f1_score_fold.append(np.mean(np.array(f1_score_model)))
            precision_fold.append(np.mean(np.array(precision_model)))
            recall_fold.append(np.mean(np.array(recall_model)))

            roc_fold.append(np.mean(np.array(roc_model)))

            mse_fold.append(np.mean(np.array(mse_model)))
            correlation_fold.append(np.mean(np.array(correlation_model)))

            # print (y_test)
            # print (y_pred.shape)
            # print (y_pred)
            mismatch_count = zero_one_loss(y_test2.T, y_pred, normalize=False)  # measures count of mismatch based on
            # zero-one-loss i.e. entire set of labels must be correct else loss for that sample = 1

            accuracy_fold.append(np.mean(np.array(accuracy_model)))
            distance_fold.append(1.0 - float(
                mismatch_count / len(X_test)))  # measure accuracy for each label, hence distance based on hamming loss
        # self.output_message += "predicted Y : %s"%y_pred + "\n"
        # self.output_message += "original Y : %s"%y_test + "\n"

        # take average of metrics at fold level and append in sequence
        # ['accuracy','f1_weighted','precision_weighted', 'recall_weighted','roc_auc']
        cv_scores.append(np.mean(np.array(accuracy_fold)))
        cv_scores.append(np.mean(np.array(f1_score_fold)))
        cv_scores.append(np.mean(np.array(precision_fold)))
        cv_scores.append(np.mean(np.array(recall_fold)))
        cv_scores.append(np.mean(np.array(roc_fold)))
        reg_scores.append(np.mean(np.array(mse_fold)))
        reg_scores.append(np.mean(np.array(correlation_fold)))
        # print("reg_scores=")
        # print(reg_scores)
        f2_score = np.mean(np.array(f2_score_fold))
        if classifier_type < 2.0:  # svm classifier
            SV_Tree = 0.0001 if np.isnan(np.nanmean(np.array(sv_list_fold))) else np.nanmean(np.array(sv_list_fold))
        # else:
        # 	SV_Tree = 0.0001
        # print("SV_tree")
        # print(SV_Tree)
        Dist_Metric = np.mean(np.array(distance_fold))
        # print(Dist_Metric)
        if SV_Tree < 0 or SV_Tree > 1:
            SV_Tree = 0
        return cv_scores, SV_Tree, f2_score, Dist_Metric, reg_scores

    '''# function to calculate remaining goal values (as per the fitness function) for each of the individual solutions
     in a given population goal_values inialised to zero and calculated in this function
    # This function is called by evaluate_indivdiuals() for each individual solution. final_selected is features post
     intersection with GA.
    # classifier_type - is the value of the 1st position of the individual solution that decides the classifier type'''

    def train_Classifier(self, dataTran, final_selected, classifier_type, C_val, Gamma_val, TreeCount, finalFlag, lrate,
                         epochs, batchsize, layers, neurons, optimizer, dropout):
        inputs = []  # initialise X of the model
        Colindices = []  # feature indices selected
        reg_scores = []
        outputs = np.ravel(np.array(deepcopy(self.class_Labels)).T)  # assign to a variable so that
        # synthetic data labels can be appended
        # Create classification model dataset aligned with features selected
        # X i.e. input to be of the form shape [n_samples, n_features]
        # dataTRan is np.ndarray with shape as required by X
        for feature in range(len(final_selected)):
            if final_selected[feature] == 1:  # append only those feature that are 1 i.e. selected
                Colindices.append(feature)  # get the indices of features selected

        # print (Colindices)
        self.output_message += "Feature indices in final selection : %s" % Colindices + "\n"
        # append all rows i.e. observations for the feature selected
        inputs = dataTran[:, Colindices]  # type = numpy array
        # print("Input dataset created for selected features with shape : ")
        # print(inputs.shape)
        #######################################################################
        # ##################### Choose the classifier ##########################
        # print("Training classifiers...")

        SV_Tree_neurons = 0.0001  # assign insignificantly low value so that goal value is not zero
        Sum_neurons = 0.0001
        Dist_Metric = 0.0001
        if classifier_type < 1.0:  # linear svm
            clf = svm.SVR(C=C_val, kernel='linear')  # probability to compute ROC
        elif 1.0 <= classifier_type < 2.0:  # RBF svm
            clf = svm.SVR(C=C_val, gamma=Gamma_val, kernel='rbf')
        elif 2.0 <= classifier_type < 3.0:  # random forest, n_jobs=-1 for parallelism
            clf = RandomForestRegressor(n_estimators=int(TreeCount), n_jobs=4)
            SV_Tree_neurons = float((self.max_trees - int(TreeCount)) / self.max_trees)  # don't round so that
            # insignificantly low values are kept
        else:  # Neural network
            if layers < 1:
                layers = 1
            elif 1 <= layers < 2:
                layers = 2
            elif 2 <= layers < 3:
                layers = 3
            else:
                layers = 4

            if neurons < 1:
                neurons = 32
            elif 1 <= neurons < 2:
                neurons = 64
            elif 2 <= neurons < 3:
                neurons = 128
            elif 3 <= neurons < 4:
                neurons = 256
            else:
                neurons = 512
            sum_of_neurons = 0
            for i in range(layers):
                sum_of_neurons += neurons
            clf = self.neural_model(lrate, inputs.shape[1], layers, neurons, optimizer, dropout)

        # ### flag to check if this is for final model save or evaluation of solutions during GA
        if finalFlag:  # if final model being trained then no CV required          
            if classifier_type < 3.0:
                clf.fit(inputs, outputs)  # train on all data
            else:
                # 3d vectors
                inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
                clf.fit(inputs, outputs, batch_size=batchsize, epochs=epochs, verbose=0)
            # print("final model fitting done")            
            return clf, inputs, outputs

        # print ("doing cvs")
        # print (np.array(outputs).shape)
        #######################################################################################
        # ##################### Run CV and get the performance scores ##########################
        # performs stratified Kfold for classification prob, so ratio of classes is maintained in folds
        skf = KFold(n_splits=self.getFolds())  # stratified to preserve the percentage of samples for
        # each class.

        if classifier_type < 3.0:
            cv_scores = cross_validate(clf, inputs, outputs, cv=skf, scoring=self.score_types, n_jobs=4)
        else:
            # 3d vectors
            inputs_neural = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
            outputs_neural = outputs
            clf_class = KerasRegressor(self.neural_model,
                                       learning_rate=lrate,
                                       input=inputs.shape[1],
                                       layers=layers,
                                       neurons=neurons,
                                       opt=optimizer,
                                       drop=dropout,
                                       epochs=epochs,
                                       batch_size=batchsize
                                       )
            # skfold = StratifiedKFold(n_splits=self.getFolds())
            cv_scores = cross_validate(clf_class, inputs_neural, outputs_neural, cv=skf, scoring=self.score_types,
                                       n_jobs=1, verbose=0)

        # get SVs count for RBF and linear SVM
        # if classifier_type <2.0: #Linear and RBF svm
        # compute dist metric for all classifiers and SVs for SVM
        # skf = KFold(n_splits=self.getFolds())  # stratified to preserve the percentage of samples for
        # each class.
        sv_list = []
        neurons_list = []
        dist_list = []
        mse_fold = []
        correlation_fold = []
        for train_index, test_index in skf.split(inputs, outputs):  # takes 20% as test data
            X_train, X_test = inputs[train_index, :], inputs[test_index, :]
            # inputs is ndarray so use np.ndarray slicing
            y_train, y_test = np.array(outputs)[train_index], np.array(outputs)[test_index]
            if classifier_type < 3.0:
                clf.fit(X_train, y_train)  # fit on train data
            else:
                # 3d vectors
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                clf.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, verbose=1)
            train_size = (len(X_train))
            y_test_reg = copy.deepcopy(y_test)
            for f in range(len(y_test_reg)):
                if y_test_reg[f] > 0:
                    y_test[f] = 1
                else:
                    y_test[f] = 0

            if classifier_type < 2.0:
                sv_size = np.sum(clf.n_support_)
                # self.output_message += "\n Support Vectors count : %s"%(sv_size) + "\n"
                # print ("support vectors count%s"%sv_size)
                if train_size > 0:
                    sv_list.append(float((train_size - sv_size) / train_size))  # don't round
            elif 3 <= classifier_type < 4.0:
                if train_size > 0:
                    neurons_list.append(float((self.max_neurons * layers*train_size -
                                               self.get_total_number_of_neurons(clf, False)) / self.max_neurons * layers*train_size))  # don't round
            # compute distance metric
            y_pred = clf.predict(X_test)
            y_pred_reg = copy.deepcopy(y_pred)
            y_pred_reg = np.reshape(y_pred_reg, (y_pred_reg.shape[0]))
            for f in range(len(y_pred_reg)):
                if y_pred_reg[f] > 0:
                    y_pred[f] = 1
                else:
                    y_pred[f] = 0
            mse_fold.append(1.0 / (1 + mean_squared_error(y_test_reg.T, y_pred_reg)))
            correlation_fold.append(st.spearmanr(y_test_reg.T, y_pred_reg))

            err = 0.0
            for err_index in range(len(X_test)):
                err = err + abs(y_pred[err_index] - y_test[err_index])
            # store the k-fold error and accuracy of the ensemble model
            dist_list.append(1.0 - float(err / len(X_test)))
        # dist_list.append(err/len(X_test))

        if classifier_type < 2.0:
            SV_Tree_neurons = float(np.mean(np.array(sv_list)))
        elif 3 <= classifier_type < 4.0:
            SV_Tree_neurons = float(np.mean(np.array(neurons_list)))

        if SV_Tree_neurons < 0 or SV_Tree_neurons > 1:
            SV_Tree_neurons = 0
        '''if Sum_neurons < 0 or Sum_neurons > 1:
            Sum_neurons = 0'''

        Dist_Metric = np.mean(np.array(dist_list))
        reg_scores.append(np.mean(np.array(mse_fold)))
        reg_scores.append(np.mean(np.array(correlation_fold)))
        return cv_scores, SV_Tree_neurons, Dist_Metric, reg_scores

    '''#This function is used to calculate the optimization goals and is called by biomarker_feature_select()
    # Inputs are single individual solution, pVal for features calculated from wilcoxon & transpose data for Kbest &
     MI feature selection
    # goal_values initialized to zero for single sol and calculated in this function'''

    def evaluate_individuals(self, individuals, pval, dataTran, goal_values, ftwo_scorer, JMIMScores, MRMRScores,
                             post_evaluate, kBestFeatFinal, multilabel):
        # print ("Indv into evaluate")
        # print (len(individuals))
        if len(individuals) == 0:  # test for parallel processing hanging without error
            with open('ERROREvalInd.txt', 'w') as feval:  # redirect all print statements to file output
                print("Empty ind into evaluate individuals", file=feval)

            goal_values[0] = 0.0001
            goal_values[1] = 0.0001
            goal_values[2] = 0.0001
            goal_values[3] = 0.0001
            goal_values[4] = 0.0001
            goal_values[5] = 0.0001
            goal_values[6] = 0.0001
            goal_values[7] = 0.0001
            goal_values[8] = 0.0001
            goal_values[9] = 0.0001
            goal_values[10] = 0.0001
            # goal_values = self.getGoalValues()

            return goal_values

        feature_count = len(self.proteins)  # this is same as len(self.dataset) as dataset is proteins in rows

        # iterate on each individual to get feature selection methods and its parameters
        significant_features = 0  # counter for initial feature selection
        number_of_selected_feature = 0  # counter for final feature selection
        # create different lists within list for feature length
        selected = [0 for _ in range(feature_count)]  # initialise list to hold features selected from univariate method
        final_selected = [0 for _ in range(
            feature_count)]  # final list to hold features selected from GA intersect univariate method

        # ######################## Check if NO features selected by GA #################################
        skip_flag = False
        GA_feature = individuals[self.evaluate_values:len(individuals)]
        if max(GA_feature) <= self.threshold:
            # print("all zeros or less than 0.5 for feature in GA. length is: %s" % len(GA_feature))
            skip_flag = True  # don't compute next steps of feature selection if no features selected by GA
            final_selected.extend([0])  # to avoid ValueError: Writing 0 cols but got 428 aliases
            if post_evaluate:  # else returns goals
                return final_selected
            # explicitly set insignificantly low value so that sol gets pushed to end of pareto front
            goal_values[0] = 0.0001
            goal_values[1] = 0.0001
            goal_values[2] = 0.0001
            goal_values[3] = 0.0001
            goal_values[4] = 0.0001
            goal_values[5] = 0.0001
            goal_values[6] = 0.0001
            goal_values[7] = 0.0001
            goal_values[8] = 0.0001
            goal_values[9] = 0.0001
            goal_values[10] = 0.0001
            # goal_values = self.getGoalValues()

        ################################################################################################################
        # ########################## choose and compute the univariate & multi-variate feature selection method#########
        ################################################################################################################
        if not skip_flag:
            # position 3 holds the feature selection method
            if individuals[3] < 1.0:  # wilcoxon rank method
                # filter the selected features as per p-value threshold. selected =1; not selected = 0
                # features start from position after the optimisation parameters, hence start from that pos
                for i in range(self.evaluate_values,
                               len(individuals)):  # iterate on each gene or feature of a given individual solution
                    # print (i-self.evaluate_values)
                    if pval[i - self.evaluate_values] < individuals[4]:
                        # select features if pval for any label < threshold

                        selected[i - self.evaluate_values] = selected[i - self.evaluate_values] + 1
                        significant_features += 1

            # KBest Method
            if 1.0 <= individuals[3] < 2.0:
                # run Kbest as per kbest feature selection value at position 5, counting from 0
                kBestSelected = kBestFeatFinal[
                    int(individuals[5]) - 1]  # get Kbest features; feature starts from 1 so -1 to get correct index
                # kBestSelected = kBestFeatFinal[0] # for testing'

                for i in range(self.evaluate_values,
                               len(individuals)):  # iterate on each gene or fetaure of a given individual solution
                    if int(kBestSelected[i - self.evaluate_values]) == 1:  # select features selected for any label
                        selected[i - self.evaluate_values] = selected[i - self.evaluate_values] + 1
                        significant_features += 1

            # MI Method - Joint Mutual Information Maximization
            if 2.0 <= individuals[3] < 3.0:
                # run MI as per neighbours value at position 6
                # JMIM k value is from 2 thru 10, so do list (index - 2) to access right values

                MIScore = JMIMScores[int(individuals[6]) - 2]
                # MIScore = JMIMScores[0] # for testing rollback to prior one

                for i in range(self.evaluate_values,
                               len(individuals)):  # iterate on each gene or feature of a given individual solution
                    # filter the selected features as per MI threshold. selected =1; not selected = 0
                    if int(MIScore[i - self.evaluate_values]) == 1:  # MI threshold is at position 4, counting from 0
                        selected[i - self.evaluate_values] = selected[i - self.evaluate_values] + 1
                        significant_features += 1
            # np.savetxt("JMIMFeaturesSelected.csv", selected, delimiter=",")

            # MRMR Method - MI based; Max. Relevance & Min. Redundancy
            if 3.0 <= individuals[3] <= 4.0:
                # run MI as per neighbours value at position 6
                # MRMR k value is from 2 thru 10, so do list (index - 2) to access right values
                MIScore = MRMRScores[int(individuals[6]) - 2]
                # MIScore = MRMRScores[0] # fr testing rollback to prior

                for i in range(self.evaluate_values, len(individuals)):  # iterate on each gene or feature of a given
                    # individual solution
                    # filter the selected features as per MI threshold. selected =1; not selected = 0
                    if int(MIScore[i - self.evaluate_values]) == 1:  # MI threshold is at position 4, counting from 0
                        selected[i - self.evaluate_values] = selected[i - self.evaluate_values] + 1
                        significant_features += 1
            # np.savetxt("MRMRFeaturesSelected.csv", selected, delimiter=",")
            # #########################################################################################################
            # Pick the features selected as intersection of common features selected between Univariate method and GA
            # #########################################################################################################

            # selected list index starts from 0 and individuals list index start from self.evaluate_values i.e.
            # length of optimisation parameters
            # needs to be checked
            for i in range(self.evaluate_values, len(individuals)):
                # if feature value is > threshold i.e. 0.5 then its selected for GA
                if selected[i - self.evaluate_values] == 1:
                    if i - self.evaluate_values in indices:
                        individuals[indices[0] + self.threshold] = sum(
                            individuals[j + self.evaluate_values] for j in indices) / len(indices)
                        if individuals[indices[0] + self.threshold] > self.threshold:
                            number_of_selected_feature = number_of_selected_feature + 1
                            final_selected[i - self.evaluate_values] = final_selected[i - self.evaluate_values] + 1
                    if individuals[i] > self.threshold:
                        number_of_selected_feature = number_of_selected_feature + 1
                        final_selected[i - self.evaluate_values] = final_selected[i - self.evaluate_values] + 1
                else:
                    final_selected[i - self.evaluate_values] = final_selected[i - self.evaluate_values] + 0
                    # to avoid ValueError: Writing 0 cols but got 428 aliases
            # print (final_selected)

            # for i in range(self.evaluate_values, len(individuals)):
            # # if feature value is > threshold i.e. 0.5 then its selected for GA
            # #todos : Activate again feature selection with other methods
            # #if selected[i-self.evaluate_values] == 1 and individuals[i] > self.threshold:
            # if individuals[i] > self.threshold:
            # number_of_selected_feature = number_of_selected_feature + 1
            # final_selected[i-self.evaluate_values] = final_selected[i-self.evaluate_values] + 1
            # else:
            # final_selected[i-self.evaluate_values] = final_selected[i-self.evaluate_values] + 0 # to avoid
            # ValueError: Writing 0 cols but got 428 aliases
            # #print (final_selected)

            # print("Initial Count of Features selected : %s" % significant_features)
            # print("Final Count of Features selected : %s" % number_of_selected_feature)
            if post_evaluate:
                return final_selected

            # print("Feature selection completed")
            # update goal value for each individual as per count of features selected
            # position 0 - complexity i.e. inverse of number of features selected.
            if number_of_selected_feature == 0:  # explicitly set insignificantly low value so that sol gets pushed
                # to end of pareto front
                goal_values[0] = 0.0001
                goal_values[1] = 0.0001
                goal_values[2] = 0.0001
                goal_values[3] = 0.0001
                goal_values[4] = 0.0001
                goal_values[5] = 0.0001
                goal_values[6] = 0.0001
                goal_values[7] = 0.0001
                goal_values[8] = 0.0001
                goal_values[9] = 0.0001
                goal_values[10] = 0.0001
            # goal_values = self.getGoalValues()
            # print(goal_values)

            if number_of_selected_feature > 0:  # values initialised to zero, so updated if only selected feature >0
                goal_values[0] = (1.0 / (1 + number_of_selected_feature))  # don't round so that insignificantly low
                # values don't become zero

            if number_of_selected_feature > 0:  # values initialised to zero, so updated if only selected feature >0
                # goal_values.(i-self.evaluate_values)
                # print('\nCalculating remainder goal values for the individual solution')
                # position 0 of inidvidual solution holds the type of classifier to run i.e. linear SVM, RBF , Random
                # forest
                # position 1 is C value for linear SVM and position 2 is gamma for RBF
                # position 7 is levels and position 8 is trees for random forest
                # set finalFlag to False, this ensures CV for evaluation
                finalFlag = False
                # global variable not read in parallelism, so take deep copy
                if len(multilabel) == 1:  # for Single label
                    # print("Single Label Run")
                    cv_scores, sv_tree, Dist_Metric, reg_scores = self.train_Classifier(
                        dataTran, final_selected, individuals[0], individuals[1], individuals[2], individuals[8],
                        finalFlag, individuals[9], int(individuals[10]), int(individuals[11]),
                        individuals[12], individuals[13], individuals[14], individuals[15])

                else:  # train classifier chain for multi label
                    # print("Multi Label Run")
                    cv_scores, sv_tree, f2_score, Dist_Metric, reg_scores = self.train_Classifier_MultiLabel(
                        copy.deepcopy(dataTran), copy.deepcopy(final_selected), copy.deepcopy(individuals[0]),
                        copy.deepcopy(individuals[1]), copy.deepcopy(individuals[2]), copy.deepcopy(individuals[7]),
                        copy.deepcopy(individuals[8]), copy.deepcopy(ftwo_scorer), copy.deepcopy(finalFlag),
                        copy.deepcopy(multilabel), copy.deepcopy(individuals[9]), copy.deepcopy(int(individuals[10])),
                        copy.deepcopy(int(individuals[11])), copy.deepcopy(individuals[12]),
                        copy.deepcopy(individuals[13]),
                        copy.deepcopy(individuals[14]), copy.deepcopy(individuals[15]))
                # print("Classifiers have been trained")
                # self.goal_header = "Feature,Accuracy,F1_weighted, F2_weighted, Precision_weighted,Recall_weighted,
                # Roc_auc,SVs-Trees,Distance,Weighted_Sum,Classification_Model,MSE,correlation"
                # assign goal values from CV scores # don't round so that insignficantly low values don't become zero
                if len(multilabel) == 1:  # for Single label
                    goal_values[1] = 0.0001 if np.mean(cv_scores['test_neg_mean_squared_error']) == 0.0 else \
                        1 / (1 - np.mean(cv_scores['test_neg_mean_squared_error']))
                    goal_values[2] = 0.0001 if np.mean(cv_scores['test_r2']) == 0.0 else np.mean(
                        cv_scores['test_r2'])
                    # goal_values[3] = 0.0001 if np.mean(f2_score) == 0.0 else np.mean(f2_score)
                    goal_values[3] = 0.0001 if np.mean(cv_scores['test_neg_median_absolute_error']) == 0.0 else \
                        1 / (1 - np.mean(cv_scores['test_neg_median_absolute_error']))
                    goal_values[4] = 0.0001 if np.mean(cv_scores['test_explained_variance']) == 0.0 else np.mean(
                        cv_scores['test_explained_variance'])

                    goal_values[7] = 0.0001 if reg_scores[0] == 0.0 or np.isnan(reg_scores[0]) else reg_scores[0]
                    goal_values[8] = 0.0001 if reg_scores[1] == 0.0 or np.isnan(reg_scores[1]) else reg_scores[1]
                else:  # for multi-label cv_score is list and not dictionary
                    goal_values[1] = 0.0001 if cv_scores[0] == 0.0 else cv_scores[0]
                    goal_values[2] = 0.0001 if cv_scores[1] == 0.0 else cv_scores[1]
                    goal_values[3] = 0.0001 if cv_scores[2] == 0.0 else cv_scores[2]
                    goal_values[4] = 0.0001 if cv_scores[3] == 0.0 else cv_scores[3]

                    goal_values[7] = 0.0001 if reg_scores[0] == 0.0 or np.isnan(reg_scores[0]) else reg_scores[0]
                    goal_values[8] = 0.0001 if reg_scores[1] == 0.0 or np.isnan(reg_scores[1]) else reg_scores[1]

                goal_values[5] = 0.0001 if sv_tree == 0.0 else sv_tree
                goal_values[10] = individuals[0]  # classification model selected
                # goal_values[9] = 0.0001 if Sum_neurons == 0.0 else Sum_neurons
                goal_values[6] = 0.0001 if Dist_Metric == 0.0 else Dist_Metric
                # print ("length of denominator for weighted sum of goals")
                # print (len(goal_values[ind]) - 2)
                # weighted sum ; minus 2 to exclude weighted sum and classification  method column
                goal_values[9] = (float(goal_values[0]) * float(self.goal_significance[0]) +
                                  float(goal_values[1]) * float(self.goal_significance[1]) +
                                  float(goal_values[2]) * float(self.goal_significance[2]) +
                                  float(goal_values[3]) * float(self.goal_significance[3]) +
                                  float(goal_values[4]) * float(self.goal_significance[4]) +
                                  float(goal_values[5]) * float(self.goal_significance[5]) +
                                  float(goal_values[6]) * float(self.goal_significance[6]) +
                                  float(goal_values[7]) * float(self.goal_significance[7]) +
                                  float(goal_values[8]) * float(self.goal_significance[8])) / (len(goal_values) - 2)

                if goal_values[9] == 0.0:
                    goal_values[9] = 0.0001

        return goal_values

    # This function is used by non_Dominated_Sol() to compare
    # if a sol i.e. goal_value dominates another sol i.e. goal_value
    def dominate(self, solution1, solution2):
        check = 2  # initialise solution comparison check to 2 i.e. equal
        ffs = len(solution1)
        dominate1 = 1
        equal1 = 1
        f = 0
        while f < ffs and dominate1 == 1:
            if solution1[f] > solution2[f]:
                equal1 = 0
            elif solution1[f] == solution2[f]:
                equal1 = 1
            else:
                dominate1 = 0
            f = f + 1

        if dominate1 == 1 and equal1 == 0:
            check = 1
        elif dominate1 == 1 and equal1 == 1:
            check = 2
        else:
            dominate2 = 1
            equal2 = 1
            f = 0
            while f < ffs and dominate2 == 1:
                if solution2[f] > solution1[f]:
                    equal2 = 0
                elif solution2[f] == solution1[f]:
                    do_nothing = 1
                else:
                    dominate2 = 0
                f = f + 1

            if dominate2 == 1 and equal2 == 0:
                check = 3

        return check

    # function to bucket solutions of a given generation into diff. pareto fronts
    # This function is called by biomarker_feature_select()
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # A dominates B iff it is better than or equal to B on all dimensions and strictly better on at least one dimension
    # def non_Dominated_Sol(self,population,goal_values,individuals):
    # function inherited from ensemble_multiclassifier.py for Missense
    def getParetoFront(self, population, goal_values, stopIndx):
        assigned = 0
        fronts = [0] * population  # initialise pareto front for population size
        front = 1
        #  create copy of goal_values so that original goal are not changed. Exclude weighted average
        eval_temp = copy.deepcopy(np.array(goal_values)[:, :stopIndx:1])  # change asarray to array to create copy
        eval_temp = eval_temp.T  # take transpose so that goals are in rows and observations in columns
        # print("Pareto front")
        # eval_goal = copy.deepcopy(goal_values) # take deep copy
        # eval_temp = list(map(list, zip(*eval_goal))) # create transpose of list of lists i.e. goals
        # print("Number of fitness values: %d" % len(eval_temp))
        # print("Population size: %d" % len(eval_temp[0]))

        self.Pareto_Wheel_Output += "Goal length in pareto front {}: \n".format(len(eval_temp))

        number_of_solutions = copy.deepcopy(population)  # deep copy of length of population
        self.Pareto_Wheel_Output += "Solution length in pareto front {}: \n".format(number_of_solutions)

        # ref: Mishra, K. K., & Harit, S. (2010). A fast algorithm for finding the non-dominated set in multi
        # objective optimization. International Journal of Computer Applications, 1(25), 35-39.
        while assigned < population:  # iterate until non-dominated solutions assigned is less than population size
            non_dominated_solutions = [0] * number_of_solutions
            index = [i for i in range(len(eval_temp[0]))]  # generate list of index based on count of goals
            self.Pareto_Wheel_Output += "Index for Ordered List {}: \n".format(index)

            eval_temp_index = list(zip(eval_temp[0], index))  # create list of tuples with goal val of sol. & index
            # sort the goal values in decrease order i.e. highest comes 1st
            self.Pareto_Wheel_Output += "Tuple of goal and list index {}: \n".format(eval_temp_index)
            ordered_list = sorted(range(len(eval_temp_index)), key=lambda k: eval_temp_index[k], reverse=True)
            self.Pareto_Wheel_Output += "Ordered list for Pareto Front {}: \n".format(ordered_list)

            non_dominated_solutions[0] = ordered_list[0]  # assign the 1st goal values as 1st non-dominated sol.
            number_of_non_dominated_solutions = 1  # initialise the non-dominated sol count
            self.Pareto_Wheel_Output += "Non-dominated solution for Pareto Front {}: \n".format(
                non_dominated_solutions)

            for i in range(1, number_of_solutions):  # iterate from 1 to pop size as 1st i.e. 0 index is already
                # selected
                n = 0
                condition = 0
                condition2 = 1
                # print ("in loop")
                while n < number_of_non_dominated_solutions and condition == 0:  # compare ordered list of solutions
                    # pairs
                    solution1 = [0] * (len(eval_temp))  # excludes weighted average goal & classification model
                    solution2 = [0] * (len(eval_temp))  # excludes weighted average goal & classification model

                    for j in range(len(eval_temp)):  # iterate on no of goals; compare solutions by goals
                        solution1[j] = eval_temp[j][ordered_list[i]]
                        solution2[j] = eval_temp[j][ordered_list[n]]

                    check = self.dominate(solution1, solution2)  # compare the goal values in descending order
                    if check == 3:
                        condition = 1
                        condition2 = 0
                    elif check == 1:
                        if number_of_non_dominated_solutions == 1:
                            condition = 1
                            non_dominated_solutions[0] = ordered_list[i]
                        else:
                            number_of_non_dominated_solutions = number_of_non_dominated_solutions - 1
                            del non_dominated_solutions[n]

                    n = n + 1

                if condition2 == 1:
                    non_dominated_solutions[number_of_non_dominated_solutions] = ordered_list[i]
                    number_of_non_dominated_solutions = number_of_non_dominated_solutions + 1

            sorted_non_dominated_solutions = sorted(non_dominated_solutions,
                                                    reverse=True)  # index sorting of non-dominated sol
            self.Pareto_Wheel_Output += "Sorted non-dominated solution index {} \n".format(
                sorted_non_dominated_solutions)
            self.Pareto_Wheel_Output += "Non-dominated solutions: {} \n".format(non_dominated_solutions)
            for i in range(number_of_non_dominated_solutions):
                assigned = assigned + 1
                # if fronts[sorted_non_dominated_solutions[i]] == 0:
                fronts[sorted_non_dominated_solutions[i]] = front
                for j in range(len(eval_temp)):  # sets the chosen goals to zero
                    # self.Pareto_Wheel_Output += "Setting to zero goals for %s "%(eval_temp[j]
                    # [sorted_non_dominated_solutions[i]]	)  + "\n"
                    eval_temp[j][sorted_non_dominated_solutions[i]] = -1000
            front = front + 1

        self.Pareto_Wheel_Output += "Calculated Pareto Frontiers {} \n\n".format(fronts)
        # print("Pareto Frontiers Calculation Completed")

        # ## check if front = 0 for error capture
        if min(fronts) == 0:
            # f = open('FrontERROR.txt','w') # redirect all print statements to file output
            # print ('Pareto front zero detected',file=f)
            # assign front 0 to be max(front)+1 so that it is worst sol
            newFront = max(fronts) + 1
            for i, f in enumerate(fronts):
                if f == 0:
                    # fronts[i]=max(fronts)+1
                    fronts[i] = newFront
        # self.Pareto_Wheel_Output += "After Zero Front Reassignment: Calculated Pareto Frontiers %s "%(fronts)
        # + "\n\n"
        # np.savetxt(path + os.sep +'ZEROFRONTOutput.txt', [self.Pareto_Wheel_Output], fmt='%s', newline='\n')
        # print Pareto and Roulette Wheenl message

        return fronts  # returns list of pareto fronts for the observations i.e. population

    # Tune fitness values by locating and using solution niches
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # uses distance for solutions within a given pareto front to see if they are similar or diff
    # and assigns a new fitness score
    def tuneFitnessValue(self, fronts, sigma_share, evaluation_values, rep, population, individuals):
        # ###### evaluation values has last values as weighted average
        min_values = self.min_values
        max_values = self.max_values

        # print("Starting Fitness Tuning As Per Pareto Frontiers")

        for i in range(1, max(
                fronts) + 1):  # start from 1st pareto front; index of fronts is starting with 1, increment by 1
            self.Pareto_Wheel_Output += "Fitness Tuning As Per Pareto Front: {} \n".format(i)
            ind = [y for y, x in enumerate(fronts) if x == i]  # get individuals in pareto front
            self.Pareto_Wheel_Output += "Individuals in pareto front {} \n".format(ind)

            # Calculate max values per goal per pareto frontier
            max_significances = [-1000] * (len(evaluation_values) - 1)  # exclude weighted average
            for j in range(len(ind)):
                for goal in range(len(evaluation_values) - 1):
                    if evaluation_values[goal][ind[j]] >= max_significances[goal]:
                        max_significances[goal] = evaluation_values[goal][ind[j]]

            self.Pareto_Wheel_Output += "Max. Significance {} \n".format(max_significances)

            # compute distance based similarity for the solutions within a given front
            for j in range(len(ind)):  # iterate for individuals within a given front
                m = 0
                for k in range(len(ind)):
                    d = 0
                    # gene_count=0
                    for gene in range(len(individuals[0])):
                        d = d + ((individuals[ind[j]][gene] - individuals[ind[k]][gene]) / float(
                            max_values[gene] - min_values[gene])) ** 2
                    # gene_count= gene_count + 1
                    d = math.sqrt(d / (len(individuals[0])))

                    if d <= sigma_share:
                        m = m + (1 - ((d / float(sigma_share)) ** 2))
                    if m == 0:
                        m = 1
                self.Pareto_Wheel_Output += "Value of m  {} \n".format(m)
                for goal in range(len(evaluation_values) - 1):
                    evaluation_values[goal][ind[j]] = float(max_significances[goal]) / m

        # # recalculated weighted sum of goals using goal_significance
        # goals are in rows and samples in columns
        for i in range(len(evaluation_values[0])):  # compute weighted sum for each sample
            evaluation_values[-1][i] = 0  # reset weighted average i.e. last row to zero
            for j in range(len(evaluation_values) - 1):  # exclude last row i.e. weighted sum in iteration
                evaluation_values[-1][i] = evaluation_values[-1][i] + evaluation_values[j][i] * self.goal_significance[
                    j]
            evaluation_values[-1][i] = evaluation_values[-1][i] / float(len(evaluation_values) - 1)

        # # Update Sum of weighted avg goal and avg of weighhted avg goal for the generation
        # to be used for visualisation
        sum_ranked = 0
        for i in range(population):
            sum_ranked = sum_ranked + evaluation_values[-1][i]

        self.sum_ranked_eval_per_generation[rep] = sum_ranked
        self.average_ranked_eval_per_generation[rep] = sum_ranked / population

        # #### write tuned goals to csv for testing for 3 generations only
        # if rep <4:
        #   df = pd.DataFrame(evaluation_values)
        #   df.to_csv(path_or_buf = pathFeature + os.sep+ str(rep)+"GoalsTunedPostFront.csv", index=False, header=False)

        # print("Fitness Tuning As Per Pareto Frontiers Completed")

        return evaluation_values

    # roulette wheel based selection of ranked individuals
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # evaluation values are of shape goals X samples
    def roulette_Wheel(self, evaluation_values_tuned, population, individuals, rep, bestSolIndex):

        min_values = self.min_values
        selected_individuals = [[0 for x in range(len(min_values))] for x in range(population)]
        indv_wheel = copy.deepcopy(
            individuals)  # included this to avoid any link between individuals and selected indv.
        # calculate the cumulative proportions i.e. selection probability based on weighted avg goal
        sum_prop = [0] * (population + 1)
        for i in range(1, population + 1):
            sum_prop[i] = sum_prop[i - 1] + evaluation_values_tuned[-1][i - 1] / float(
                self.sum_ranked_eval_per_generation[rep])

        self.Pareto_Wheel_Output += "Cumulative proportions based on weighted avg post tuning of Fitness " \
                                    "Value: {}\n".format(sum_prop)

        for i in range(1, population):  # get proportions from index 1 through pop size as 0 is for best ind.
            random_number = random.uniform(0, 1)  # generate probability randomly
            for j in range(0, population):  # select those indv. for which random no is within cumulative prop
                if sum_prop[j] <= random_number < sum_prop[j + 1]:
                    selected_individuals[i] = copy.deepcopy(indv_wheel[j])  # assign via deepcopy and not original

        # # assign best sol based on max of weighted sum of goals to index 0
        selected_individuals[0] = copy.deepcopy(indv_wheel[bestSolIndex])  # assign via deepcopy to avoid index editing

        # print("Roulette Wheel Selection Completed")

        return selected_individuals

    # function to apply two-point cross over
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    def GACross_Over(self, population, cross_indv):

        # print("Doing Two Point Crossover")
        self.output_message += "Crossover population {}\n".format(population)

        temp_individuals = [[0 for x in range(len(self.min_values))] for x in range(population)]

        self.output_message += "Temp individual length {}\n".format(len(temp_individuals))

        cross_indv_copy = copy.deepcopy(cross_indv)  # take deep copy to avoid index 0 editing
        features = copy.deepcopy(len(cross_indv_copy[0]))  # take deep copy to avoid index 0 editing

        best_sol_before = sum(cross_indv[0])  # index 0 holds best sol
        self.output_message += "best sol sum before {}\n".format(sum(cross_indv[0]))

        for i in range(1, population - 1, 2):  # 1st is best sol so preserve it
            random_number = random.uniform(0, 1)  # generate prob randomly
            if random_number < self.twoCross_prob:
                self.output_message += "Doing 2-point crossover for sol: {}, {}\n".format(i, i + 1)
                cross_point1 = 0
                cross_point2 = 0
                while cross_point1 == cross_point2:
                    cross_point1 = math.ceil((features - 4) * random.uniform(0, 1))  # minus 4 to leave last positions
                    if cross_point1 < math.floor((2 * features - 1) / 3):
                        width = math.ceil(random.uniform(0, 1) * (math.floor(features - 1) / 3 - 2))
                        cross_point2 = cross_point1 + width
                    else:
                        width = math.ceil(random.uniform(0, 1)) * (
                                math.floor(features / 3 - 1) - 2 - (cross_point1 - math.floor(2 * features / 3)))
                        cross_point2 = cross_point1 + width
                if cross_point1 > cross_point2:
                    temp_cross_point = cross_point1
                    cross_point1 = cross_point2
                    cross_point2 = temp_cross_point
                width = int(width)
                cross_point1 = int(cross_point1)
                cross_point2 = int(cross_point2)
                for j in range(cross_point1, cross_point2 + 1):
                    temp_individuals[i][j] = copy.deepcopy(cross_indv_copy[i + 1][j])
                for j in range(cross_point1, cross_point2 + 1):
                    cross_indv_copy[i + 1][j] = copy.deepcopy(cross_indv_copy[i][j])
                for j in range(cross_point1, cross_point2 + 1):
                    cross_indv_copy[i][j] = copy.deepcopy(temp_individuals[i][j])

            elif self.twoCross_prob <= random_number < (self.twoCross_prob + self.arithCross_prob):
                # arithmetic cross over
                self.output_message += "Doing arithmetic crossover for sol: {}, {}\n".format(i, i + 1)
                alpha = random.uniform(0, 1)
                for j in range(0, features):
                    temp_individuals[i][j] = copy.deepcopy(
                        alpha * cross_indv_copy[i][j] + (1 - alpha) * cross_indv_copy[i + 1][j])
                    temp_individuals[i + 1][j] = copy.deepcopy(
                        (1 - alpha) * cross_indv_copy[i][j] + (alpha) * cross_indv_copy[i + 1][j])
                for k in range(0, features):
                    cross_indv_copy[i][k] = copy.deepcopy(temp_individuals[i][k])
                    cross_indv_copy[i + 1][k] = copy.deepcopy(temp_individuals[i + 1][k])

        best_sol_after = sum(cross_indv_copy[0])  # index 0 holds best sol; take after sum from copy version
        self.output_message += "best sol sum after {}\n".format(sum(cross_indv_copy[0]))

        # ## check if best solution is preserved
        if (best_sol_before - best_sol_after) != 0.0:
            with open('CrossOverERROR.txt', 'w') as fcross:  # redirect all print statements to file output
                self.output_message += "Cross over of best sol detected \n"
                # print('Cross over for best solution detected. Sol sum delta %s' % (best_sol_before - best_sol_after),
                #       file=fcross)

        # print("Crossover completed")
        return copy.deepcopy(cross_indv_copy)

    # function to apply adaptive mutation; Use deep copy for assignment to avoid index 0 editing
    # adapted from script written for Missense SNPs ensemble_multiclassifier.py by Konstantinos Theofilatos
    # script edited for adaptive mutation rate
    def adaptiveMutation(self, population, mutate_indv, rep):
        # print("Doing Adaptive Mutation")
        min_values = self.min_values
        max_values = self.max_values
        self.output_message += "Doing adaptive mutation \n"
        self.output_message += "population size {}\n".format(population)

        best_sol_before = sum(mutate_indv[0])  # index 0 holds best sol
        mutate_indv_copy = copy.deepcopy(mutate_indv)  # take deep copy so that index 0 is not chnaged

        # get the mutation probability and gaussian variance proportion based on average similarity
        # ref: Rapakoulia, T., Theofilatos, K., Kleftogiannis, D., Likothanasis, S., Tsakalidis, A., & Mavroudi, S.
        # (2014).
        # EnsembleGASVR: a novel ensemble method for classifying missense single nucleotide polymorphisms.
        # Bioinformatics, 30(16), 2324-2333.

        # compute average similarity between the members of the population and its best member i.e. at index 0
        a = np.array(mutate_indv_copy[:0:-1])  # take all elements from bottom to 1, exclude 0
        b = np.array([mutate_indv_copy[0]])  # best solution
        CDist = distance.cdist(a, b, 'chebyshev')
        AvgSimilarity = np.mean(CDist) / (np.max(CDist) - np.min(CDist))
        self.Pareto_Wheel_Output += "Avg. Similarity {} \n".format(AvgSimilarity)

        numerator = float(self.mutation_prob[1]) - (1 / self.population)  # 0.2-(1/Pop size)
        delta = float(rep) * numerator / float(self.generations)
        if round(AvgSimilarity, 2) >= round(self.avgSimThr,
                                            2):  # high similarity, so increase the diversity with higher mutation
            mutationRate = float(self.mutation_prob[1]) + delta  # 0.2+ delta
            gaussVarProp = float(self.gaussMut_varPro[1]) + delta  # 0.5+ delta
        else:
            mutationRate = float(self.mutation_prob[1]) - delta  # 0.2 - delta
            gaussVarProp = float(self.gaussMut_varPro[1]) - delta  # 0.5 - delta
            if mutationRate < float(self.mutation_prob[0]):  # reset if lesser than lower bound
                mutationRate = float(self.mutation_prob[0])
            if gaussVarProp < float(self.gaussMut_varPro[0]):  # reset if lesser than lower bound
                gaussVarProp = float(self.gaussMut_varPro[0])

        self.Pareto_Wheel_Output += "Mutation Rate chosen {} \n".format(mutationRate)
        self.Pareto_Wheel_Output += "Gaussian Variance Proportion chosen {} \n".format(gaussVarProp)

        feature = copy.deepcopy(len(mutate_indv_copy[0]))
        # apply mutation operator
        for i in range(1, population):  # preserve the 1st sol as its best
            for j in range(0, feature):
                random_number = random.uniform(0, 1)
                if random_number < mutationRate:
                    # Gaussian distribution. mean zero and standard deviation = (max-min)*variance proportion
                    mutate_indv_copy[i][j] = copy.deepcopy(
                        mutate_indv_copy[i][j] + random.gauss(0, gaussVarProp * (max_values[j] - min_values[j])))
            # self.output_message += "sol mutated: %s"%i + "\n"

            # Correct values out of boundaries
            for j in range(0, len(min_values)):
                if mutate_indv_copy[i][j] < min_values[j]:
                    mutate_indv_copy[i][j] = copy.deepcopy(min_values[j])
                if mutate_indv_copy[i][j] > max_values[j]:
                    mutate_indv_copy[i][j] = copy.deepcopy(max_values[j])

        best_sol_after = sum(
            mutate_indv_copy[0])  # index 0 holds best sol; take sum of index 0 from copy to check if any change

        # print(best_sol_before - best_sol_after)
        # ## check if best solution is preserved
        if (best_sol_before - best_sol_after) != 0.0:
            with open('MutationERROR.txt', 'w') as fmutate:  # redirect all print statements to file output
                self.output_message += "Mutation of best sol in Gen {} \n".format(rep)
                # print('Mutation of best solution detected. Sol sum delta %s' % (best_sol_before - best_sol_after),
                #       file=fmutate)

        return copy.deepcopy(mutate_indv_copy)

    def labelPowerset(self):
        # take transpose of list of list i.e. the class labels outcome Y of shape [n_labels, n_sample]
        yNew = list(map(list, zip(*self.class_Labels)))  # yNew shape [n_samples, n_labels]
        # convert to list of strings
        yNew = [[str(int(j)) for j in i] for i in yNew]  # change int to str
        self.powerSetClass = list(map(''.join, yNew))  # list of strings with powerset of class labels given the dataset

    # return yNewStr # list of strings with powerset of class labels given the dataset

    '''#This function pre-computes the wilcoxon ranksum, JMIM and MRMR for given set of k values.
    # it then passes diff. populations of teh generation for evaluation to evaluate_individuals()
    # Pareto optimal solution search and cross-over + mutation for GA is also done here'''

    def biomarker_feature_select(self):

        # clock the time taken for feature selection
        start = time.perf_counter()  # start the clock to capture time
        self.output_message += "Parameters selected Gen: {} , Pop: {} , Sampling: {} \n".format(
            self.generations, self.population, self.ImbalanceDataSample)
        # capture the parameters
        # if len(MULTI_LABELS) == 1:  # single label problem
        self.classification_problems = len(self.dictCountLabels)  # count of classes
        self.output_message += "Single Label, count of classes : {}\n".format(self.classification_problems)
        # print("Count of classes : %s" % self.classification_problems)
        '''else:
            self.labelPowerset()  # transform Y to Label powerset i.e. Y = 'Y1Y2Y3' etc
            self.output_message += "Multi-Label, count of powerset classes : %s" % (len(set(self.powerSetClass))) + "\n"
            self.output_message += "Multi-Label, Powerset classes created: %s" % (set(self.powerSetClass)) + "\n"
            # print("new labels with powerset labels %s" % len(set(self.powerSetClass)))
            # print("original classes %s" % len(self.class_Labels_binary))'''

        # #############################Apply Wilcoxon Rank Sum feature selection #####################################
        pVal = self.wilcoxon_rank_sum_test()
        # print("Wilcoxon Rank Sum Completed")

        # dataset as np.array and transpose. To be used by Kbest and MI feature selection methods
        # data format for multi-label class : X (n_samples, n_features); y (n_samples, n_labels)
        new_data = np.array([np.array(x) for x in self.dataset])
        dataTran = np.transpose(new_data)

        # ########################### SelectKbest features, min feature = 1 required #########
        # KBestScores = [] # list to hold kBest per feature count from 1 to 400 i.e. min to max as specified in main
        kBestFeatFinal = [0 for _ in range(len(self.proteins))]
        kcount = 0
        # print("Computing SelectKBest ")
        # print("Total number of proteins : %d" % len(self.proteins))
        # lbl = copy.deepcopy(MULTI_LABELS)
        lbl = [1]
        # compute selectKbest in parallel
        kBestFeatFinal = Parallel(n_jobs=self.num_threads)(delayed(
            self.kbest_features)(dataTran, k, kBestFeatFinal, lbl) for k in range(int(self.min_values[5]),
                                                                                  int(self.max_values[5] + 1.0)))
        # print("%d features selected from Kbest feature selection" % len(kBestFeatFinal))
        df = pd.DataFrame(kBestFeatFinal)
        df.to_csv(path_or_buf=self.pathFeature + os.sep + "KBestMultiLbl.csv", index=False, header=False)

        #  Uses parallelism; run in command prompt and not Spyder. Spyder gives error with parallelism

        #  Apply JMIM feature selection; pre-compute to use within GA
        JMIMScores = []
        kcount = 0

        # print("Computing JMIM ")

        for k in range(int(self.min_values[6]), int(self.max_values[6] + 1.0)):
            JMIMScores.append([])
            try:
                JMIMSc = self.MutualInfo_features(dataTran, k, 'JMIM')
            except ValueError as e:
                # print("JMIM ill-defined for k = %s" % k)
                # print("Selecting ZERO features")
                JMIMSc = [False for _ in range(len(self.proteins))]
            # print ("Selecting all features")
            # JMIMSc = [True for _ in range(len(self.proteins))]

            JMIMScores[kcount].extend(
                JMIMSc)  # list of list of lists - 1st nested is k and then JMIM for each label for that k
            kcount = kcount + 1

        df = pd.DataFrame(JMIMScores)
        df.to_csv(path_or_buf=self.pathFeature + os.sep + "JMIMMultiLbl.csv", index=False, header=False)

        # ############################# Apply MRMR feature selection; pre-compute to use within GA ####################
        MRMRScores = []
        kcount = 0

        # print("Computing MRMR ")
        # ## add try and exception - ill-defined k, set to 0
        for k in range(int(self.min_values[6]), int(self.max_values[6] + 1.0)):  # takes k = 2 thru 10
            MRMRScores.append([])
            # print (k)
            try:
                MRMRSc = self.MutualInfo_features(dataTran, k, 'MRMR')
            except ValueError as e:
                # print("MRMR ill-defined for k = %s" % k)
                # print("Selecting ZERO features")
                MRMRSc = [False for _ in range(len(self.proteins))]
            # print ("Selecting all features")
            # MRMRSc = [True for _ in range(len(self.proteins))]

            MRMRScores[kcount].extend(MRMRSc)
            kcount = kcount + 1

        df = pd.DataFrame(MRMRScores)
        df.to_csv(path_or_buf=self.pathFeature + os.sep + "MRMRMultiLbl.csv", index=False, header=False)  # pval data

        # #################### PERFORM OVERSAMPLING FOR IMBALANCED CLASS  ################################
        # This should be set to False because no oversampling is performed on regression NEW
        boosted = False
        if self.ImbalanceDataSample:
            # print("Imbalanced data detected. Generating Synthetic data to boost imbalanced class data")
            data_with_labels = pd.DataFrame(dataTran)
            data_with_labels['labels'] = self.class_Labels[0]
            try:
                boosted_dataset = smogn.smoter(data=data_with_labels, y='labels')
            except ValueError:
                boosted = False
        if boosted:
            boosted_labels = list(boosted_dataset.pop('labels'))

            dataTran = np.array(boosted_dataset)  # concatenate by rows this new sample with existing data

            df = pd.DataFrame(dataTran)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "OrgWithSynData.csv", index=False, header=False)
            # pval data

            self.class_Labels[0] = list(boosted_labels)

            df = pd.DataFrame(self.class_Labels)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "OrgWithSynLabels.csv", index=False, header=False)

        #################################################################################################
        # run GA based optimisation to choose feature selection method i.e. Wilcoxon, KBest, MI, mRMR or Union of these
        # and associated parameters such as p-val for Wilcoxon, neighbours for MI
        #################################################################################################

        individuals = self.initialize()  # Initialize Population of Individual Solutions
        self.output_message += "Initial population formulated. Sample individuals length : {}, {}\n".format(
            len( individuals[0]), len(individuals[4]))
        self.output_message += "Sample individual : {}\n".format(individuals[4])

        # print("\nIndividuals Initialised.")

        # variables for boundary condition / hold output of each generation
        generations = self.getGenerations()
        population = self.getPopulation()

        # make custom score f2 for cross_validate
        ftwo_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
        sigma_share = 0.5 / (float(len(individuals[0])) ** 0.1)  # sigma for distance threshold

        # Apply Evolutionary Process to Optimize Individual Solutions
        # print("Starting GA based optimisation...")
        premature_termination = 0
        front1_feat_perGen = []  # list of lists to hold front1 features per  generation
        front1_sol_perGen = []  # list of lists to hold front1 sol per  generation
        front1_index = 0  # index to write front1 sol per generation
        final_selected_best = []  # list of lists to hold best sol features per generation
        best_index = 0  # index to write best sol per generation

        solHeader = ['Model', 'C', 'Gamma', 'Feat Filter', 'P-Val', 'Feat Count', 'Neighbors',
                     'Classifier Chain', 'Tree Count', 'Learning rate', 'Epochs', 'Batch Size', 'Layers', 'Neurons',
                     'Optimizer', 'Dropout'] + self.proteins

        for rep in range(generations):  # iterate on generations
            # print("\nGeneration:" + str(rep + 1))
            # initialise goal values to zero, list of list with length = population size; calcualted by
            # evaluate_individuals()
            # position 0 - feature count complexity i.e. inverse of number of features selected. Higher value is
            # less complex
            # position 1 - Negative Mean Absolute S Error
            # position 2 - weighted R2 score
            # position 3 - Negative Median Absolute Error
            # position 4 - explained_variance
            # position 5 - support vectors selected for SVM or trees for random forest or sum of neurons for the
            # convolutional layers of the neural model
            # position 6 - manhattan distance
            # position 7 - MSE
            # position 8 - correlation
            # position 9 - weighted sum of scores
            # position 10 - numeric code of classifier selected
            goal_values = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
            # create different lists within list for feature length

            # goal_values = self.getGoalValues()
            # ########### Evaluate population of solutions.
            self.output_message += "\nGeneration : {}\n".format(rep)
            logging.debug("Generation : {}".format(rep))
            post_evaluate = False

            # evaluate individuals of a given population in parallel
            goal_values_old = copy.deepcopy(goal_values)
            goal_values = Parallel(n_jobs=2)(
                delayed(self.evaluate_individuals)(ind, copy.deepcopy(pVal), copy.deepcopy(dataTran),
                                                   copy.deepcopy(goal_values),
                                                   copy.deepcopy(ftwo_scorer), copy.deepcopy(JMIMScores),
                                                   copy.deepcopy(MRMRScores), copy.deepcopy(post_evaluate),
                                                   copy.deepcopy(kBestFeatFinal), copy.deepcopy([1])) for ind
                in individuals)

            # print("%d individuals" % len(individuals))
            # print("goal_values")
            # print(goal_values)
            # print(len(goal_values))
            # print(len(goal_values[0]))
            average_performance = 0  # average performance for convergence. initialise to zero
            convergence = 0.0
            # [i for i in goal_values if i] # this will create a new list which does not have any empty item,
            # else zip returns null if any value is null
            # avgAll = [float(sum(col))/len(col) for col in zip(*goal_values)] # avg of all goals
            avgAll = [float(sum(col)) / len(col) for col in zip(*[i for i in goal_values if i])]  # avg of all goals
            # print("Average of all goals %s" % avgAll)
            # print("Number of all goals: %d" % len(avgAll))
            average_performance = copy.deepcopy(avgAll[9])  # index 9 has weighted avg
            self.average_eval_per_generation[rep] = copy.deepcopy(average_performance)
            best_performance = list(
                map(max, zip(*[i for i in goal_values if i])))  # includes max of classification model as well
            # print()
            best_avg_perf = copy.deepcopy(best_performance[9])  # max weighted average; index 9 has weighted avg
            bestSolIndex = 0
            for idx in range(len(goal_values)):
                # if round(goal_values[idx][9],4) == round(best_avg_perf,4): # float comparison, so using round
                if goal_values[idx][9] < -1:
                    print(goal_values[idx])
                if goal_values[idx][9] == best_avg_perf:  # float comparison
                    bestSolIndex = idx
                    self.output_message += "best sol index found\n"
                    break

            self.output_message += "best sol index {}\nbest avg performance {}\n avg of avg goal {}\n".format(
                bestSolIndex, best_avg_perf, average_performance)

            convergence = math.fabs(best_avg_perf - average_performance)
            # print("\nConvergence value %s" % convergence)
            # Convergence criterion is checked in order to stop the evolution if the population is deemd us converged
            if convergence < self.convergeThreshold * average_performance:
                premature_termination = 1

            # Estimate pareto fronts for the solutions of this generation based on all goals except weighted average
            stopIndx = (len(goal_values[0]) - 2)  # goals to slice for pareto front computation, excludes weighted
            # average
            # print("\nPareto stopIndx is: %s" % stopIndx)
            self.Pareto_Wheel_Output += "Generation : {}\n".format(rep + 1)
            fronts = self.getParetoFront(len(copy.deepcopy(individuals)), goal_values, stopIndx)

            # #### save front1 solutions, features are intersect of GA individuals and filter method
            post_evaluate = True  # when true this flag avoid classifier fitting within evaluate
            for_front1 = copy.deepcopy(individuals)
            for i, ind in enumerate(for_front1[:]):  # iterate on solutions to pic sol with front = 1
                if fronts[i] == 1:
                    # print("%d" % i)
                    front1_sol_perGen.append([])
                    front1_sol_perGen[front1_index].append(rep)  # add generation number
                    front1_sol_perGen[front1_index].extend(ind)
                    front1_feat_perGen.append([])
                    front1_feat_perGen[front1_index].append(rep)  # add generation number
                    front1_feat_perGen[front1_index].extend(
                        self.evaluate_individuals(ind, copy.deepcopy(pVal), copy.deepcopy(dataTran),
                                                  copy.deepcopy(goal_values[i]), copy.deepcopy(ftwo_scorer),
                                                  copy.deepcopy(JMIMScores), copy.deepcopy(MRMRScores), post_evaluate,
                                                  copy.deepcopy(kBestFeatFinal), copy.deepcopy([1])))
                    front1_index = front1_index + 1

            # get max per goal; and best sol
            self.max_eval_per_generation[rep] = list(map(max, zip(*[i for i in goal_values if i])))[9]
            self.output_message += "best performance :" + "\n"
            self.output_message += str(self.max_eval_per_generation[rep])
            self.output_message += "\n"

            self.max_sol_per_generation[rep] = copy.deepcopy(individuals[bestSolIndex])  # use index of bestSolIndex,
            # there can be more than one index with max weighted average
            self.output_message += "best solution :\n"
            self.output_message += str(self.max_sol_per_generation[rep])
            self.output_message += "\n"
            self.output_message += str(individuals[bestSolIndex])
            self.output_message += "\n"

            # get final features for best sol per gen, features are intersect of GA individuals and filter method
            post_evaluate = True  # when true this flag avoid classifier fitting within evaluate
            final_selected_best.append([])
            # print("\nFinal features for best solution per generation")
            final_selected_best[best_index].extend(
                self.evaluate_individuals(copy.deepcopy(individuals[bestSolIndex]), copy.deepcopy(pVal), dataTran,
                                          goal_values[bestSolIndex], ftwo_scorer, copy.deepcopy(JMIMScores),
                                          copy.deepcopy(MRMRScores), post_evaluate, copy.deepcopy(kBestFeatFinal),
                                          copy.deepcopy([1])))
            best_index = best_index + 1  # increment index to write best sol per generation

            # Check if its last generation, do not apply pareto, roulette wheel, cross over and mutation
            if rep == (generations - 1) or premature_termination == 1:

                if premature_termination == 0:
                    # print("This was last generation without convergence")
                    self.output_message += "No Convergence\n"
                else:
                    # print("Solution converged")
                    self.output_message += "Convergence at Generation : {}\n".format(rep + 1)

                np.savetxt(self.pathFeature + os.sep + "goals_Final.csv", goal_values, delimiter=",",
                           header=self.goal_header)

                # get final features for final sol, features are intersect of GA individuals and filter method
                post_evaluate = True
                post_evaluate2 = False  # when true this flag avoid classifier fitting within evaluate
                finalFeat_selected = []
                for_final = copy.deepcopy(individuals)
                # ##KOSTAS
                # if len(MULTI_LABELS) == 1:
                # print("multi_label_test")
                for i, ind in enumerate(for_final[:]):
                    finalFeat_selected.append([])
                    finalFeat_selected[i].extend(
                        self.evaluate_individuals(ind, copy.deepcopy(pVal), dataTran, goal_values[i], ftwo_scorer,
                                                  copy.deepcopy(JMIMScores), copy.deepcopy(MRMRScores),
                                                  post_evaluate, copy.deepcopy(kBestFeatFinal),
                                                  copy.deepcopy([1])))

                try:
                    df = pd.DataFrame(finalFeat_selected)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "features_Final.csv", index=False,
                              header=self.proteins)
                except ValueError as e:
                    df = pd.DataFrame(finalFeat_selected)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "features_Final.csv", index=False,
                              header=False)

                # write pareto front1 of the final solution; Sol sets in Pareto front 1 to be used in prediction
                # model
                front1FinalIndv = []
                front1FinalGoal = []
                front1FinalFeature = []
                index = 0
                # for i in range(population): # iterate on solutions to pic sol with front = 1
                for_final_front1 = copy.deepcopy(individuals)
                for i, ind in enumerate(for_final_front1[:]):
                    if fronts[i] == 1 and max(finalFeat_selected[i]) > 0:
                        front1FinalIndv.append([])
                        front1FinalIndv[index].extend(ind)
                        front1FinalGoal.append([])
                        front1FinalGoal[index].extend(goal_values[i])
                        front1FinalFeature.append([])
                        front1FinalFeature[index].extend(finalFeat_selected[i])
                        index = index + 1

                try:
                    df = pd.DataFrame(front1FinalIndv)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolFront1.csv", index=False,
                              header=solHeader)
                except ValueError as e:
                    df = pd.DataFrame(front1FinalIndv)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolFront1.csv", index=False, header=False)

                np.savetxt(self.pathFeature + os.sep + "goals_FinalFront1.csv", front1FinalGoal, delimiter=",",
                           header=self.goal_header)

                # #### ValueError: Writing 0 cols but got 428 aliases
                try:
                    df = pd.DataFrame(front1FinalFeature)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "features_FinalFront1.csv", index=False,
                              header=self.proteins)
                except ValueError as e:
                    # print("Final front 1 feature of length 0 encountered")
                    # print(finalFeat_selected)
                    # print(front1FinalFeature)
                    df = pd.DataFrame(front1FinalFeature)
                    df.to_csv(path_or_buf=self.pathFeature + os.sep + "features_FinalFront1.csv", index=False,
                              header=False)
                break

            # write original sol

            # compute fitness value guided by grouping of sol in each pareto fronts and recompute weighted average
            # so retain weighted average goal
            stopIndxAvg = (len(goal_values[0]) - 1)  # last column is classification model so exclude that
            # print("stopIndxAvg %s" % stopIndxAvg)

            evaluation_values = copy.deepcopy(np.asarray(goal_values)[:, :stopIndxAvg:1])
            evaluation_values = evaluation_values.T
            evaluation_values_tuned = self.tuneFitnessValue(fronts, sigma_share, evaluation_values, rep,
                                                            len(individuals), individuals)

            # roulett wheel selection on tuned evaluation values by pareto front
            wheel_indv = copy.deepcopy(individuals)  # take a deep copy so that best sol is not altered
            selected_individuals = self.roulette_Wheel(evaluation_values_tuned, len(individuals), wheel_indv, rep,
                                                       bestSolIndex)

            # do cross-over
            cross_indv = copy.deepcopy(
                selected_individuals)  # take deep copy so that best ind is not altered; index 0 solution
            cross_individuals = self.GACross_Over(len(individuals), cross_indv)

            # do mutation
            mutate_indv = copy.deepcopy(
                cross_individuals)  # take deep copy so that best ind is not altered; index 0 solution
            mutated_individuals = self.adaptiveMutation(len(individuals), mutate_indv, rep)

            # csv writing of data for testing
            # df = pd.DataFrame(individuals)
            # df.to_csv(path_or_buf=self.pathFeature + os.sep + str(rep) + "OriginalIndvPop.csv", index=False,
            #           header=False)
            # np.savetxt(self.pathFeature + os.sep + str(rep) + "goals_Population.csv", goal_values,
            # delimiter=",", header=self.goal_header)

            # df = pd.DataFrame(selected_individuals)
            # df.to_csv(path_or_buf=self.pathFeature + os.sep + str(rep) + "SelectedIndvPop.csv", index=False,
            #           header=False)  # csv writing of data for testing

            # df = pd.DataFrame(cross_individuals)
            # df.to_csv(path_or_buf=self.pathFeature + os.sep + str(rep) + "CrossOverIndvPop.csv", index=False,
            #          header=False)  # csv writing of data for testing

            # df = pd.DataFrame(mutated_individuals)
            # df.to_csv(path_or_buf=self.pathFeature + os.sep + str(rep) + "MutatedIndvPop.csv", index=False,
            #           header=False)  # csv writing of data for testing

            # update the population with the offspings
            individuals = copy.deepcopy(mutated_individuals)

        # last solution is the final solution
        try:
            dfFinalSol = pd.DataFrame(individuals)
            dfFinalSol.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolutionsAll.csv", index=False,
                              header=solHeader)
        except ValueError as e:
            dfFinalSol = pd.DataFrame(individuals)
            dfFinalSol.to_csv(path_or_buf=self.pathFeature + os.sep + "FinalSolutionsAll.csv", index=False,
                              header=False)

        self.output_message += "\nTime (in minutes) taken for feature selection and parameters optimisation : %s" % (
                (time.perf_counter() - start) / 60.00) + "\n"
        np.savetxt(self.pathFeature + os.sep + 'FeatureSelectionOutput.txt', [self.output_message], fmt='%s',
                   newline='\n')  # print output_message to text
        np.savetxt(self.pathFeature + os.sep + 'ParetoRWheelOutput.txt', [self.Pareto_Wheel_Output], fmt='%s',
                   newline='\n')  # print Pareto and Roulette Wheenl message

        # ### write avg and best performance goal values before pareto and avg after pareto
        dfAvg = pd.DataFrame(self.average_eval_per_generation)
        dfAvg.to_csv(path_or_buf=self.pathFeature + os.sep + "AvgOfAvgGoalsPerGen.csv", index=False, header=False)

        dfAvgFront = pd.DataFrame(self.average_ranked_eval_per_generation)
        dfAvgFront.to_csv(path_or_buf=self.pathFeature + os.sep + "AvgOfAvgGoalPerGenPostFront.csv", index=False,
                          header=False)

        try:
            df = pd.DataFrame(final_selected_best)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "features_FinalBestSol.csv", index=False,
                      header=self.proteins)
        except ValueError as e:  # if convergence happens then final_selected has lesser
            df = pd.DataFrame(final_selected_best)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "features_FinalBestSol.csv", index=False, header=False)

        # front1 sol per gen
        try:
            df = pd.DataFrame(front1_sol_perGen)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "Front1SolPerGen.csv", index=False,
                      header=["Gen"] + solHeader)
        except ValueError as e:
            df = pd.DataFrame(front1_sol_perGen)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "Front1SolPerGen.csv", index=False, header=False)

        # front1 features final per gen
        try:
            df = pd.DataFrame(front1_feat_perGen)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "Front1FeatPerGen.csv", index=False,
                      header=["Gen"] + self.proteins)
        except ValueError as e:
            df = pd.DataFrame(front1_feat_perGen)
            df.to_csv(path_or_buf=self.pathFeature + os.sep + "Front1FeatPerGen.csv", index=False, header=False)

        try:
            dfMax = pd.DataFrame(self.max_eval_per_generation)
            dfMax.to_csv(path_or_buf=self.pathFeature + os.sep + "BestPerformancePerGen.csv", index=False,
                         header=False)
        except TypeError as e:
            logging.exception("typeError: object of type int has no len() occurred for best sol generation :")
            max_eval = [i for i in self.max_eval_per_generation if i]
            if len(max_eval) > 0:
                logging.info("Writing best goals by dropping null")
                dfMax = pd.DataFrame(max_eval)
                dfMax.to_csv(path_or_buf=self.pathFeature + os.sep + "BestPerformancePerGen.csv", index=False,
                             header=False)

        try:
            dfBest = pd.DataFrame(self.max_sol_per_generation)
            dfBest.to_csv(path_or_buf=self.pathFeature + os.sep + "BestSolPerGen.csv", index=False, header=solHeader)
        except TypeError as e:
            max_sol = [i for i in self.max_sol_per_generation if i]
            if len(max_sol) > 0:
                dfBest = pd.DataFrame(max_sol)
                dfBest.to_csv(path_or_buf=self.pathFeature + os.sep + "BestSolPerGen.csv", index=False, header=False)

        return dataTran, front1FinalIndv, front1FinalFeature, final_selected_best[-1]

    ##################################################################################
    # function to finalise and save FINAL Model for prediction
    def FinaliseAndSaveModel(self, dataTran, front1FinalIndv, front1FinalFeature):
        # print("Fitting and saving final model(s) for prediction")
        # ##### Finalizing the model : train the model on the entire training dataset available
        finalFlag = True  # set the flag for final solution True so that CV is not done when training the classifiers
        skip_count = 0  # check how many models with 0 features skipped
        model_count = 0  # check how many models selected
        # print(MULTI_LABELS)
        # chain_dictionary = {} # initialise dictionary for writing final front1 classiifer chain as dictionary
        chain_dictionary = []  # initialise list for writing final front1 classifier chain. List allows same chain to
        # be associated with more than one model
        # start the loop post optimisation length of optimisation parameters
        classifier_list = []
        classifier_ind = []
        X = np.array([])
        Y = np.array([])
        for i, individual in enumerate(front1FinalIndv):  # iterate on all sol selected
            # train the classifier ; front1FinalIndv[ind][0] = classifier_type; front1FinalIndv[ind][1] = C_val;
            # front1FinalIndv[ind][2] = Gamma_val; front1FinalIndv[ind][7] = Treelevel; front1FinalIndv[ind][0] =
            # TreeCount,ftwo_scorer = 0
            if len(individual) > 0:  # check for zero features
                try:
                    clf, X, Y = self.train_Classifier(copy.deepcopy(dataTran), front1FinalFeature[i], individual[0],
                                                      individual[1], individual[2], copy.deepcopy(individual[8]),
                                                      finalFlag, individual[9], int(individual[10]),
                                                      int(individual[11]), individual[12], individual[13],
                                                      individual[14], individual[15])
                    if 0 <= individual[0] < 3:
                        clf.fit(X, Y)
                    else:
                        inputs_neural = np.reshape(X, (X.shape[0], X.shape[1], 1))
                        outputs_neural = Y
                        clf.fit(inputs_neural, outputs_neural, batch_size=int(individual[11]),
                                epochs=int(individual[10]), verbose=0)

                    # Storage of Support vectors or trees generated for each model in the Front
                    with open(self.pathFeature + os.sep + "number_of_svs.txt", 'a') as num_sv_file:
                        if 0 <= individual[0] < 2:
                            sv_trees = 'Model {} - Number of Support Vectors: {}\n'.format(i + 1, int(
                                np.sum(clf.n_support_)))
                        elif 2 <= individual[0] < 3:
                            trees = int(individual[8])
                            sv_trees = 'Model {} - Number of Random Forest Trees: {}\n'.format(i + 1, trees)
                        else:
                            neurons = self.get_total_number_of_neurons(clf, False)
                            sv_trees = 'Model {} - Number of Neurons in CNN: {}\n'.format(i + 1, neurons)
                        num_sv_file.write(sv_trees)
                    if individual[0] < 3:
                        joblib.dump(clf, '{0}/{1:05d}finalSingleModel.pkl.z'.format(self.pathModel, i + 1))
                    else:  # neural network save in hdf5 format
                        clf.save("{0}/{1:05d}finalNeuralModel.hdf5".format(self.pathModel, i + 1))
                    # save the model use joblib;
                    # Ref: http://scikit-learn.org/stable/modules/model_persistence.html
                    model_count = model_count + 1
                    classifier_list.append(clf)
                    classifier_ind.append(individual[0])
                except ValueError:  # ValueError: Found array with 0 feature(s) (shape=(550, 0))
                    # while a minimum of 1 is required.
                    skip_count = skip_count + 1
                    continue

        chain_dictionary.append([1])

        self.calculate_final_metrics(classifier_list, dataTran, np.ravel(np.array(deepcopy(self.class_Labels)).T),
                                     front1FinalFeature, classifier_ind)
        df = pd.DataFrame(chain_dictionary)
        df.to_csv(path_or_buf=self.pathModel + os.sep + "classifierChain.csv", index=False, header=False)

    def calculate_final_metrics(self, clf_list, input_data, output_data, features_selected, classifier):

        cross_mse_accuracy = []
        cross_rae = []
        cross_rrse = []
        cross_r2 = []
        cross_explained_variance = []
        cross_correlation = []
        # change 1: Initialize lists to store CV predictions and true values
        all_cv_preds = []  # To store all CV predictions
        all_cv_true = []  # To store all CV true values

        # change: Save the input and output datasets
        input_data_df = pd.DataFrame(input_data)
        input_data_df.to_csv(self.pathFeature + os.sep + "input_data.csv", index=False)

        output_data_df = pd.DataFrame(output_data)
        output_data_df.to_csv(self.pathFeature + os.sep + "output_data.csv", index=False)

        # change: Save the original datasets
        original_data_df = pd.DataFrame(input_data)
        original_data_df['Output'] = output_data
        original_data_df.to_csv(self.pathFeature + os.sep + "original_data.csv", index=False)
        
        # split into the folds, and use the same for all the classifiers in order to have the best metrics
        skf = KFold(n_splits=self.getFolds())  # no stratified because regression

        fold_number = 1  # added line
        for train_index, test_index in skf.split(input_data, output_data):  # takes 20% as test data
            X_train, X_test = input_data[train_index, :], input_data[test_index, :]
            # inputs is ndarray so use np.ndarray slicing
            y_train, y_test = np.array(output_data)[train_index], np.array(output_data)[test_index]

            # change: Save train and test sets for this fold
            fold_train_data_df = pd.DataFrame(X_train)
            fold_train_data_df['Output'] = y_train
            fold_train_data_df.to_csv(self.pathFeature + os.sep + f"cv_train_data_fold_{fold_number}.csv", index=False)
            # change: Save train and test sets for this fold
            fold_test_data_df = pd.DataFrame(X_test)
            fold_test_data_df['Output'] = y_test
            fold_test_data_df.to_csv(self.pathFeature + os.sep + f"cv_test_data_fold_{fold_number}.csv", index=False)

            # added line:
            print(f"Fold {fold_number}: Training set size: {len(X_train)}, Test set size: {len(X_test)}")

            y_ensemble_pred = []
            for i, clf in enumerate(clf_list):

                # Keep only selected features for each classifier
                Colindices = []
                for feature in range(len(features_selected[i])):
                    if features_selected[i][feature] == 1:  # append only those feature that are 1 i.e. selected
                        Colindices.append(feature)  # get the indices of features selected
                # append all rows i.e. observations for the feature selected
                X_train_feats = X_train[:, Colindices]
                X_test_feats = X_test[:, Colindices]
                if 0 <= classifier[i] < 3:
                    clf.fit(X_train_feats, y_train)
                    y_pred = clf.predict(X_test_feats)

                else:
                    # reshaping for CNN input
                    X_train_feats = np.reshape(X_train_feats, (X_train_feats.shape[0], X_train_feats.shape[1], 1))
                    X_test_feats = np.reshape(X_test_feats, (X_test_feats.shape[0], X_test_feats.shape[1], 1))

                    # Fit for neural networks need more arguments
                    clf.fit(X_train_feats, y_train, batch_size=len(X_train_feats), epochs=100, verbose=0)
                    # For neural network (we need > 0.5.astype because binary classification returns 2 probabilities.)
                    y_pred = clf.predict(X_test_feats)
                    # y_pred is (X,1) and must be (X,) so we reshape.
                    y_pred = np.reshape(y_pred, (y_pred.shape[0]))

                y_ensemble_pred.append(y_pred)
            # Using Multiclass ensemble prediction, Keep Majority prediction
            y_ensemble_pred = np.array(y_ensemble_pred).T
            final_pred = []
            for predictions in y_ensemble_pred:
                # mind majority prediction
                majority = np.mean(predictions)
                final_pred.append(majority)

            # change 2: Store the predictions and true values for CV
            all_cv_preds.extend(final_pred)
            all_cv_true.extend(y_test)

            # added lines inside the KFold loop, after generating predictions for each fold
            fold_predictions_df = pd.DataFrame({'True': y_test, 'Predicted': final_pred})  # added line
            fold_predictions_df.to_csv(self.pathFeature + os.sep + f"cv_predicted_values_fold_{fold_number}.csv", index=False)  # added line
            fold_number += 1  # added line

            # TP, FP, TN, FN = self.ROC_measures(y_test, final_pred)
            cross_mse_accuracy.append(mean_squared_error(y_test, final_pred))
            cross_rae.append(sum(abs(np.array(final_pred) - np.array(y_test.T))) /
                             sum(abs(np.array(y_test) - np.array(y_test).mean())))
            cross_rrse.append(math.sqrt(sum((np.array(y_test) - np.array(final_pred)) ** 2) /
                                        sum((np.array(y_test) - np.array(y_test).mean()) ** 2)))
            # cross_r2.append(r2_score(y_test, final_pred))
            # cross_explained_variance.append(explained_variance_score(y_test, final_pred))
            # cross_correlation.append(st.spearmanr(y_test.T, final_pred))
        cross_mse_accuracy = np.mean(np.sqrt(cross_mse_accuracy))
        cross_rae = np.mean(cross_rae)
        cross_rrse = np.mean(cross_rrse)
        # cross_r2 = np.mean(cross_r2)
        # cross_explained_variance = np.mean(cross_explained_variance)
        # cross_correlation = np.mean(cross_correlation)
        cross_r2= r2_score(all_cv_true, all_cv_preds)
        cross_explained_variance= explained_variance_score(all_cv_true, all_cv_preds)
        cross_correlation, cv_p_value,= st.spearmanr(all_cv_true, all_cv_preds)

        y_ensemble_pred = []
        for i, clf in enumerate(clf_list):
            Colindices = []
            for feature in range(len(features_selected[i])):
                if features_selected[i][feature] == 1:  # append only those feature that are 1 i.e. selected
                    Colindices.append(feature)  # get the indices of features selected
            # append all rows i.e. observations for the feature selected
            X_train_all = input_data[:, Colindices]
            if 0 <= classifier[i] < 3:
                clf.fit(X_train_all, output_data)
                y_pred = clf.predict(X_train_all)
            else:
                # reshaping for CNN input
                X_train_all = np.reshape(X_train_all, (X_train_all.shape[0], X_train_all.shape[1], 1))

                # Fit for neural networks need more arguments
                clf.fit(X_train_all, output_data, batch_size=len(X_train_all), epochs=50, verbose=0)
                y_pred = clf.predict(X_train_all)
                # y_pred is (X,1) and must be (X,) so we reshape.
                y_pred = np.reshape(y_pred, (y_pred.shape[0]))

            y_ensemble_pred.append(y_pred)
        # Using Multiclass ensemble prediction, Keep Majority prediction
        y_ensemble_pred = np.array(y_ensemble_pred).T
        final_pred = []
        for predictions in y_ensemble_pred:
            # mind majority prediction
            majority = np.mean(predictions)
            final_pred.append(majority)

        # change 3: Save CV predictions and true values
        cv_predictions_df = pd.DataFrame({'True': all_cv_true, 'Predicted': all_cv_preds})
        cv_predictions_df.to_csv(self.pathFeature + os.sep + "cv_predictions.csv", index=False)

        # TP, FP, TN, FN = self.ROC_measures(output_data, final_pred)
        training_mse_accuracy = np.sqrt(mean_squared_error(output_data, final_pred))
        training_rae = (sum(abs(np.array(final_pred) - np.array(output_data))) /
                        sum(abs(np.array(output_data) - np.array(output_data).mean())))
        training_rrse = math.sqrt(sum((np.array(output_data) - np.array(final_pred)) ** 2) /
                                  sum((np.array(output_data) - np.array(output_data).mean()) ** 2))

        training_r2 = r2_score(output_data, final_pred)
        training_explained_variance = explained_variance_score(output_data, final_pred)
        #correlation = np.mean(st.spearmanr(output_data.T, final_pred))
        correlation, p_value= st.spearmanr(output_data, final_pred)


        # change 4: Save training predictions
        training_predictions_df = pd.DataFrame({'True': output_data, 'Predicted': final_pred})
        training_predictions_df.to_csv(self.pathFeature + os.sep + "training_predictions.csv", index=False)

        # Storage of Support vectors or trees generated for each model in the Front
        with open(self.pathFeature + os.sep + "metrics.txt", 'w') as metrics_file:

            metrics_file.write('Cross validation Root Mean Square Error: {:.4f}\n'.format(cross_mse_accuracy))
            metrics_file.write('Cross validation Relative Absolute Error: {:.2f} %\n'.format(cross_rae * 100))
            metrics_file.write('Cross validation Root Relative Squared Error: {:.2f} %\n'.format(cross_rrse * 100))
            metrics_file.write('Cross validation R2 (coefficient of determination) regression score: {:.2f}\n'.format(
                cross_r2))
            metrics_file.write('Cross validation Explained Variance score: {:.2f} \n'.format(cross_explained_variance))
            metrics_file.write('Cross validation Spearman Correlation: {:.2f} \n'.format(cross_correlation))
            metrics_file.write('Cross validation Spearman p-value: {} \n'.format(cv_p_value))

            metrics_file.write('Training Root Mean Square Error: {:.4f}\n'.format(training_mse_accuracy))
            metrics_file.write('Training Relative Absolute Error: {:.2f} %\n'.format(training_rae * 100))
            metrics_file.write('Training Root Relative Squared Error: {:.2f} %\n'.format(training_rrse * 100))
            metrics_file.write('Training R2 (coefficient of determination) regression score: {:.2f}\n'.format(
                training_r2))
            metrics_file.write('Training Explained Variance score: {:.2f} \n'.format(training_explained_variance))
            metrics_file.write('Training Spearman Correlation: {:.2f} \n'.format(correlation))
            metrics_file.write('Training Spearman p-value: {} \n'.format(p_value))


def readArguments(argv):
    # Get parameters based on command line input
    # https://docs.python.org/3/library/argparse.html#module-argparse # #https://docs.python.org/3/howto/
    # argparse.html#id1
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='DATA_FILENAME', help='txt file name with omics data')  # default type is str
    parser.add_argument('label', metavar='CLASS_LABELFILE', help='txt file name with class labels')
    parser.add_argument('group_features', metavar='GROUP_FEATURES', help='features data in groups')
    parser.add_argument('-p', '--percentage', metavar='MISS_PERCENTAGE', nargs='?', type=float, default=0.8,
                        help='percentage missing data threshold')
    parser.add_argument('-k', '--neighbour', metavar='KNN_LOF_NEIGHBOURS', nargs='?', type=int, default=20,
                        help='neighbours for knn impute and LOF')
    parser.add_argument('-n', '--normalization_method', nargs='?', default='0',
                        help='normalisation method, default is MinMax scaling')
    parser.add_argument('--missing_imputation_method', nargs='?', default='2',
                        help='missing data impute method, default is knn')
    parser.add_argument('-pp', '--population', nargs='?', type=int, default=50, help='Population size, default is 50')
    parser.add_argument('-g', '--generations', nargs='?', type=int, default=200,
                        help='Count of max generations, default is 200')
    parser.add_argument('-f', '--folds', nargs='?', type=int, default=5,
                        help='Count of folds for cross validation, default is 5')
    parser.add_argument('-ft', '--fold_type', nargs='?', type=int, default=0,
                        help='Type of cross validation, 0 indicates training with the k-1 folds and testing with 1 fold'
                             ' while 1 indicates the opposite. Default is 0')
    parser.add_argument('-s', '--sampling', default=True, action='store_false',
                        help='Do sampling for imbalanced class, True when not specified')
    parser.add_argument('-t', '--trainPercentage', nargs='?', type=float, default=0.8,
                        help='Percentage of sample data for training, default is 0.8')
    parser.add_argument('-fv', '--goal_values', nargs=11, type=float, default=0.0001,
                        help='Fitness values for each solution after evaluation. Default values are 0.0001 for 11 goal'
                             ' values')

    args = parser.parse_args()  # get the mandatory and default arguments

    return args


# ################## main called when ensemble_cvd.py is run
def main(varDict, user='unknown', jobid=0, pid=0):
    # assign min-max values for below 8 parameters: Used by evaluate_individuals() including specific index reference
    # to below
    # 0 -0.001 to 4 - <1 SVM linear, >1 & <2 RBF; >2 & <3 Random forest; >3 Neural Network
    # 1 - 0.001 to 1000 for SVM C parameter
    # 2 - 0.001 to 1000 for gamma SVM RBF
    # 3 - 0.001 to 4 choice of feature selection method: <1 for wilcoxon, >=1 & <2 Kbest, >=2 & <3 MI-JMIM, >=3
    # & <4 MI-mRMR
    # 4 - 0.001 to 0.5 for wilcoxon rank p-value feature threshold i.e. reject feature with less than threshold value.
    # 0.5 allows no feature filter
    # 5 - 1.0 to 100 for Kbest & MIFS to pass count of feature selection i.e. return max 100 best features
    # 6 - 2.00 to 10 for MIFS neighbours; default MIFS neighbours is 3
    # 7 - 0.00 to 1.0 for number of permutations of the classifier chain; 0 signifies orignal sequence as read from
    # omics.txt
    # 8 - 10 to 100 for number of trees in random forest
    # 9 - 0.001 to 0.05 for learning rate in neural network
    # 10 - 50 to 200 for epochs in neural network
    # 11 - 32 to 128 for batch size in  neural network
    # 12 - 0 to 4 - <1 1 layer; >1 & <2 2 layers; >2 & <3 3 layers; >3 4 layers in neural network
    # 13 - 0 to 5 - <1 32 neurons; >1 & <2 64 neurons; >2 & <3 128 neurons; >3 & <4 256 neurons ; >4 512 neurons
    # in neurol network
    # 14 - 0 to 4 - <1 Adam; >1 & <2 RMSProp; >2 & <3 SGD; >3 NAdam in neural network
    # 15 - 0.2 to 0.5 for Dropout possibility
    min_values = [0.001, 0.001, 0.001, 0.001, 0.0001, 1.0, 2.0, 0.0, 10.0, 0.001, 50, 32, 0, 0, 0, 0.2]
    max_values = [3.999, 500.0, 500.0, 2.0, 0.5, 30.0, 5.0, 1.0, 100.0, 0.05, 200, 128, 4, 5, 4, 0.5]
    # min_values=[0.001,0.001,0.001,0.001,0.0001,50.0,5.0, 0.0,20.0] # for testing; single k and feature
    # max_values=[3.0,1000.0,1000.0,4.0,0.5,51.0,6.0, 0.0,30.0] # for testing; single k and feature
    goal_values = [0.0001 for i in range(11)]

    path = varDict['output_folder']  # get cwd and append the file path to it
    # create output folder for pre-processing with timestamp to the path

    output_folder = path + 'Output_PreProcess' + os.sep  # get the path separator based on os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pathProcess = output_folder

    # create output folder for feature Selection with timestamp to the path
    output_folder = path + 'Output_FeatSelect' + os.sep  # get the path separator based on os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pathFeature = output_folder

    # create output folder for Model Saving with timestamp to the path
    output_folder = path + 'classification_models' + os.sep  # get the path separator based on os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pathModel = output_folder

    # ################ pre-process the dataset file
    # create instance of class preProcess with non-default parameters
    process = PreProcess(varDict['file'], varDict['label'], varDict['group_features'], varDict['percentage'],
                         varDict['neighbour'], varDict['normalization_method'], varDict['missing_imputation_method'])
    process.setPath(pathProcess)
    # final data is outlierdata.csv ; dataset_initial is the original omics.txt data read by converttoarray() to be
    # used for association rule
    dataset, proteins, class_LabelNew, dataset_initial = process.biomarker_discovery_modeller()

    # f = open(pathProcess + os.sep + 'main_output.txt', 'w')  # redirect all print statements to file output

    # ##################### Create dictionary of permutations and set the min-max values #######################
    # codify string to number in multi-labels
    # MULTI_LABELS holds the original sequence as derived from omics data
    label_permutation = list(itertools.permutations(MULTI_LABELS, len(MULTI_LABELS)))
    max_values[7] = (len(label_permutation) - 1)  # set the max value; -1 as index starts from 0
    dict_chain = {}
    for i, perm in enumerate(label_permutation):
        dict_chain[i] = perm
    # print("Permutation of classifier chains %s" % dict_chain, file=f)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tPermutation of classifier chains {}".format(pid, jobid, user, dict_chain))

    if len(MULTI_LABELS) == 1:  # single label problem
        max_values[7] = 1  # set the max value to 1 as min/max can't be same else tuning of fitness value will have
    # division by zero

    # ##################### COMPUTE CLASS SPECIFIC  VARIABLES #######################
    total_boosted = []  # initialise boosting count per imbalanced class within a label

    if len([1]) == 1:  # single label problem
        # for lbl,key in enumerate(MULTI_LABELS): # get boosting count per imbalanced class within a label
        # key = MULTI_LABELS[0]  # access the 1st key
        lbl = 0  # access the 1st label
        # print("Label key %s" % key)
        dictCountLabels = dict(Counter(class_LabelNew[lbl]))
        # print("dictCountLabels %s" % dictCountLabels, file=f)
        dictRatioLabels = {k: float(dictCountLabels[k]) / float(len(dataset[0])) for k in dictCountLabels for k in
                           dictCountLabels}  # change dict values to ratios
        # change class labels to int
        dictRatioLabels = {int(k): v for k, v in dictRatioLabels.items()}
    else:
        # use powerset to find imbalanced classes
        # take transpose of list of list i.e. the class labels outcome Y of shape [n_labels, n_sample]

        yNew = list(map(list, zip(*class_LabelNew)))  # yNew shape [n_samples, n_labels]
        # convert to list of strings
        yNew = [[str(int(j)) for j in i] for i in yNew]  # change int to str
        powerSetClass = list(map(''.join, yNew))  # list of strings with powerset of class labels given the dataset
        dictCountLabels = dict(Counter(powerSetClass))
        # print("dictCountLabels %s" % dictCountLabels, file=f)
        dictRatioLabels = {k: float(dictCountLabels[k]) / float(len(dataset[0])) for k in dictCountLabels for k in
                           dictCountLabels}  # change dict values to ratios
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tdictCountLabels {}".format(pid, jobid, user, dictCountLabels))

    Ratiovalues = list(dictRatioLabels.values())  # hold ratio values in list for min compute
    # print("Ratiovalues %s" % Ratiovalues, file=f)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tRatiovalues {}".format(pid, jobid, user, Ratiovalues))

    # ###### imabalnce threshold = 1/ count of classes
    ImbalanceThreshold = round(float(1.0 / len(dictCountLabels)), 2)
    # print("imbalance threshold is %s" % ImbalanceThreshold, file=f)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\timbalance threshold is {}".format(pid, jobid, user, ImbalanceThreshold))
    # ############### if ratio of any class <1/count of classes then its imbalanced class #####
    if min(Ratiovalues) < ImbalanceThreshold:
        total = sum(dictCountLabels[k] for k in dictCountLabels)  # total records
        for i, val in enumerate(Ratiovalues):  # check which class has data less than threshold
            if val < ImbalanceThreshold:
                classKey = list(dictRatioLabels.keys())[list(dictRatioLabels.values()).index(val)]
                # count = (ImbalanceThreshold*(sum of existing records) - count of records for this class
                # (val))/ (1-ImbalanceThreshold)
                sample_count = round((ImbalanceThreshold * total - dictCountLabels[classKey]) / (
                        1 - ImbalanceThreshold))  # how may samples to generate such that ratio reached threshold
                total_boosted.append(sample_count)
            else:
                # print("Balanced data")
                total_boosted.append(0.0)  # add 0 count for boosting

    # print(total_boosted, file=f)  # list of record count to boost e.g. by label [254, 228, 104, 110]
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tTotal Boosted {}".format(pid, jobid, user, total_boosted))

    # for RF to provide similarly scaled results as SVM fix max trees to be 80% of training samples
    max_values[8] = int(float(len(dataset[0])))
    '''if not varDict['sampling']:
        max_values[8] = int(float(len(dataset[0]) * varDict['trainPercentage']))
    else:
        if len(MULTI_LABELS) == 1:  # single label problem
            max_values[8] = int(float(len(dataset[0]) + max(total_boosted)) * varDict['trainPercentage'])
        else:
            max_values[8] = int(float(len(dataset[0]) + sum(total_boosted)) * varDict['trainPercentage'])'''

    max_trees = max_values[8]  # max trees to be used for performance measure of random forest
    # print("max trees %s" % max_trees, file=f)
    # print("sampling flag is %s" % varDict['sampling'], file=f)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tmax trees {}, sampling flag is {}".format(pid, jobid, user, max_trees,
                                                                                     varDict['sampling']))

    # f.close()  # close the output file
    # ######################### perform adaptive GA based feature selection
    evaluate_values = len(min_values)

    # pass dataset and dataTran to feature selection as ranksum uses dataformat of shape dataset while others use
    # dataTran format
    feature = FeatureSelection(dataset, proteins, min_values, max_values, class_LabelNew, evaluate_values,
                               max_trees, varDict['population'], varDict['generations'], varDict['folds'],
                               goal_values, varDict['sampling'], dictCountLabels, dictRatioLabels,
                               Ratiovalues, ImbalanceThreshold, dict_chain, varDict['mutation_prob'],
                               varDict['arith_crossver_prob'], varDict['two_point_crossover_prob'],
                               varDict['thread_num'])
    feature.setGoalSignificancesByUser(varDict['feature_significance'], varDict['accuracy_significance'],
                                       varDict['model_complexity_significance'])
    feature.setPaths(pathFeature, pathModel)
    try:
        dataTran, front1FinalIndv, front1FinalFeature, feature_bestSol_final = feature.biomarker_feature_select()
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in Model creation".format(pid, jobid, user))
        return False
    # ########### create association rule for best sol i.e. index = 0 for front1FinalFeature using dataset_initial
    # feature.AssociationRule(dataset_initial,feature_bestSol_final) # written in predict file
    # ########################## save the final model for prediction
    feature.FinaliseAndSaveModel(dataTran, front1FinalIndv, front1FinalFeature)
    return True


def main_only_training(outfolder, dataset, labels, features, population, generations, folds, mutation_prob,
                       arith_crossover_prob, two_point_crossover_prob, thread_num, goal_significance_list,
                       user='unknown', jobid=0, pid=0):
    """
    Main function for creating Ensemble prediction models for a given training dataset and labels
    :param outfolder: Folder for the output files
    :param dataset: input dataset array with samples and their features
    :param labels: input labels array with true label for each sample
    :param features: input features array
    :param population: Population for GASVR
    :param generations: Generation for GASVR
    :param folds: Folds used for training dataset
    :param mutation_prob: mutation probability fro GASVR
    :param arith_crossover_prob: arithmetic crossover probability for GASVR
    :param two_point_crossover_prob: two point crossover probability for GASVR
    :param thread_num: number of available treads used for Parallel processes
    :param goal_significance_list: Goal Weight Significance for Feature,NegativeMSE,R2, Negative_MedianAbsoluteError,
    explained_variance, SVs-Trees,RMSE,correlation
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return: True if finished successfully, else False
    """

    # assign min-max values for below 8 parameters: Used by evaluate_individuals() including specific index reference
    # to below
    # 0 -0.001 to 4 - <1 SVM linear, >1 & <2 RBF; >2 & <3 Random forest; >3 Neural Network
    # 1 - 0.001 to 1000 for SVM C parameter
    # 2 - 0.001 to 1000 for gamma SVM RBF
    # 3 - 0.001 to 4 choice of feature selection method: <1 for wilcoxon, >=1 & <2 Kbest, >=2 & <3 MI-JMIM, >=3
    # & <4 MI-mRMR
    # 4 - 0.001 to 0.5 for wilcoxon rank p-value feature threshold i.e. reject feature with less than threshold value.
    # 0.5 allows no feature filter
    # 5 - 1.0 to 100 for Kbest & MIFS to pass count of feature selection i.e. return max 100 best features
    # 6 - 2.00 to 10 for MIFS neighbours; default MIFS neighbours is 3
    # 7 - 0.00 to 1.0 for number of permutations of the classifier chain; 0 signifies original sequence as read from
    # omics.txt
    # 8 - 10 to 100 for number of trees in random forest
    # 9 - 0.001 to 0.05 for learning rate in neural network
    # 10 - 50 to 200 for epochs in neural network
    # 11 - 32 to 128 for batch size in  neural network
    # 12 - 0 to 4 - <1 1 layer; >1 & <2 2 layers; >2 & <3 3 layers; >3 4 layers in neural network
    # 13 - 0 to 5 - <1 32 neurons; >1 & <2 64 neurons; >2 & <3 128 neurons; >3 & <4 256 neurons ; >4 512 neurons
    # in neurol network
    # 14 - 0 to 4 - <1 Adam; >1 & <2 RMSProp; >2 & <3 SGD; >3 NAdam in neural network
    # 15 - 0.2 to 0.5 for Dropout possibility  
    max_kbestvalue = len(features) if len(features) <= 29 else 30
    min_values = [0.001, 0.001, 0.001, 0.001, 0.0001, 1.0, 2.0, 0.0, 10.0, 0.001, 50, 32, 0, 0, 0, 0.2]
    max_values = [2.999, 500.0, 500.0, 2.0, 0.5, max_kbestvalue, 5.0, 1.0, 200.0, 0.05, 200, 128, 4, 5, 4, 0.5]
    # min_values=[0.001,0.001,0.001,0.001,0.0001,50.0,5.0, 0.0,20.0] # for testing; single k and feature
    # max_values=[3.0,1000.0,1000.0,4.0,0.5,51.0,6.0, 0.0,30.0] # for testing; single k and feature
    goal_values = [0.0001 for _ in range(11)]

    path = outfolder  # get cwd and append the file path to it

    # create output folder for feature Selection with timestamp to the path
    output_folder = path + 'feature_selection' + os.sep  # get the path separator based on os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pathFeature = output_folder

    # create output folder for Model Saving with timestamp to the path
    output_folder = path + 'classification_models' + os.sep  # get the path separator based on os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pathModel = output_folder
    # single label problem
    max_values[7] = 1  # set the max value to 1 as min/max can't be same else tuning of fitness value will have
    # division by zero

    # ##################### COMPUTE CLASS SPECIFIC  VARIABLES #######################
    total_boosted = []  # initialise boosting count per imbalanced class within a label
    dict_chain = {1: labels[0][0]}
    # single label problem
    # for lbl,key in enumerate(MULTI_LABELS): # get boosting count per imbalanced class within a label
    # key = MULTI_LABELS[0]  # access the 1st key
    lbl = 0  # access the 1st label
    dictCountLabels = dict(Counter(labels[lbl]))
    dictRatioLabels = {k: float(dictCountLabels[k]) / float(len(dataset[0])) for k in dictCountLabels for k in
                       dictCountLabels}  # change dict values to ratios
    # change class labels to int
    dictRatioLabels = {int(k): v for k, v in dictRatioLabels.items()}

    logging.info("PID:{}\tJOB:{}\tUSER:{}\tdictCountLabels {}".format(pid, jobid, user, dictCountLabels))

    Ratiovalues = list(dictRatioLabels.values())  # hold ratio values in list for min compute
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tRatiovalues {}".format(pid, jobid, user, Ratiovalues))

    # ###### imbalance threshold = 1/ count of classes
    ImbalanceThreshold = round(float(1.0 / len(dictCountLabels)), 2)
    # print("imbalance threshold is %s" % ImbalanceThreshold, file=f)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\timbalance threshold is {}".format(pid, jobid, user, ImbalanceThreshold))

    # for RF to provide similarly scaled results as SVM fix max trees to be 80% of training samples
    max_values[8] = int(float(len(dataset[0])))

    max_trees = max_values[8]  # max trees to be used for performance measure of random forest
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tmax trees {}".format(pid, jobid, user, max_trees))

    # ######################### perform adaptive GA based feature selection
    evaluate_values = len(min_values)

    # pass dataset and dataTran to feature selection as ranksum uses dataformat of shape dataset while others use
    # dataTran format
    feature = FeatureSelection(dataset, features, min_values, max_values, labels, evaluate_values,
                               max_trees, population, generations, folds, goal_values, True, dictCountLabels,
                               dictRatioLabels, Ratiovalues, ImbalanceThreshold, dict_chain, mutation_prob,
                               arith_crossover_prob, two_point_crossover_prob, thread_num)
    feature.setGoalSignificancesByUserList(goal_significance_list)
    feature.setPaths(pathFeature, pathModel)
    try:
        dataTran, front1FinalIndv, front1FinalFeature, feature_bestSol_final = feature.biomarker_feature_select()
    except ValueError:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in Model creation, maybe preprocess the input file".format(
            pid, jobid, user))
        return False, 'Error in data parsing. Maybe the dataset needs preprocessing'
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError in Model creation".format(pid, jobid, user))
        return False, 'Error in Model creation'

    feature.FinaliseAndSaveModel(dataTran, front1FinalIndv, front1FinalFeature)
    return True, ''


if __name__ == '__main__':
    args = readArguments(sys.argv[1:])  # Get parameters based on input
    varDict = vars(args)  # gives dictionary of variable name and value; NOT Namespace
    main(varDict)  # pass the dictionary of arguments to main

# run the code using command below
#
# python ensemble_CVD_MultiLabel.py omics.txt AllLabels.txt
# #### for testing python ensemble_CVD_MultiLabel.py omics.txt AllLabels.txt -g 5 -pp 50

# python ensemble_cvd.py omics.txt DiabetesLabel.txt
# ####### for testing python ensemble_CVD_MultiLabel.py omics.txt DiabetesLabel.txt -g 5 -pp 50 -s [for sampling false]
# ################ python ensemble_CVD_MultiLabel.py omics.txt DiabetesLabel.txt -g 3 -pp 20 [for sampling true]
# python ensemble_CVD_MultiLabel2.py vaccine01_dataset.txt vaccine_labels.txt group_features_new.txt -g 2 -pp 10 -s
# python ensemble_CVD_MultiLabel1.py testdata.txt ILI_any.txt group_features.txt -g 2 -pp 10


# 1) grouped feature selection later
# 2) κωδικοποίηση κατηγορικών μεταβλητών moved to backend script
# 3) διόρθωση boosting για regression (δεν λειτουργεί όπως είναι τώρα)
# 4) validation in all folds together, ΟΚ
# 5) writing output files in date folders should change. make it similar to the previous implementation, ok
# 6) multiple core execution bug: explore whether the code is functional when running it in multiple cores for high
# sample size datasets (>2000 samples)
