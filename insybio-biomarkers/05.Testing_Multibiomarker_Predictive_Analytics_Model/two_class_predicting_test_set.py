'''
two_class_predicting_test_set.py

A script to predict the labels of a test set based on a given model produced by the modeller on
a training set.

Input:
       testSet (String): the test set with only the features selected by the modeller included
       testSet_labels (String): the test set labels
       maximums_filename (String): the file with the maximum values of the testset features
       minimums_filename (String): the file with the minimum values of the testset features
       averages_filename (String): the file with the average values from the preprocessing
       features_filename (String): the selected features filename taken from the training step
       missing_imputation_method (Integer): the missing imputation method done in the preprocessing step of the training data (1, 2)
       normalization_method (Integer): the normalization method done in the preprocessing step of the training data (1, 2)
       model (String): the model created by the training step
       data_been_preprocessed_flag (Integer): 1 if it has been preprocessed, 0 if it hasn't
       variables_for_normalization_string: (String) the selected variables for normalization as a string with comma or newline separated strings, eg. "ABC,DEF,GHI"
       markers_filename (String): the names of the biomarkers as a tab separated file, one line
       filetype (Integer): 7 if it is a gene expressions file, other number if it's a biomarkers file
       has_features_header (Integer): 1 if the testSet has a features header, 0 if it doesn't
       has_samples_header (Integer): 1 if the testSet has a samples header, 0 if it doesn't
       training_labels_filename (String): the filename of the training labels
       length_of_features_filename (String): the filename with the length of the features of the training set, taken from step 02.
       selected_comorbidities_string: (String) the selected comorbidities in the step 04 that have been deleted from the original dataset, taken from step 02
Output:
        result_labels.txt: the predicted labels by the model
        metrics.txt: the metrics of the prediction

Example of calling the code:
python3.6 two_class_predicting_test_set.py testing_testing/test_dataset.txt testing_testing/test_labels.txt  Input/twoclass/new/new2/maximums.txt Input/twoclass/new/new2/minimums.txt Input/twoclass/new/new2/averages_for_missing_values_imputation.txt  Input/twoclass/new/features_list.txt 1 2 Input/twoclass/model_2018_10_03 0 "" Input/twoclass/new/markers.txt 8 0 0 testing_testing/training_labels.txt length_of_features.txt "Feature_0,Feature_1"

'''

import math
import copy
import time
import sys
import os
import csv
import logging
import random
import scipy
import numpy as np
from svmutil import *
from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

def parsing_csv(test_file):
    """
    Reading a csv data file and inserting the lines into an array,
    thus returning a list of lists.

    Args:
        test_file: a csv data file

    Returns:
        data: a list of lists
    """
    data = []
    with open(test_file,'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

def parsing_dataset(dataset_filename):
    """
    Parsing a TSV file.

    Args:
        dataset_filename: csv file to be parsed

    Returns:
        dataset: a list of lists
    """
    dataset = list()
    dataset_file = open(dataset_filename, 'r')
    number_of_lines = 0
    for line in dataset_file:
        dataset.append([])
        words = line.split("\t")
        for w in words:
            try:
                dataset[number_of_lines].append(float(w.rstrip()))
            except:
                dataset[number_of_lines].append(-1000) # to catch '' strings and timestamp data
        number_of_lines += 1
    logging.info("Dataset parsed successfully.")
    return dataset

def parsing_oneline(oneline_filename):
    """
    Parses a file with one line, with values separated with tabs. Also works with data separated with newlines for some reason.

    Args:
        oneline_filename: a file with one line

    Returns:
        maximums: a list
    """
    maximums = list()
    maximums_file = open(oneline_filename,'r')
    for line in maximums_file:
        word = line.split("\t")
        for w in word:
           maximums.append(w.rstrip())
    return maximums


def parsing_oneline_generic(oneline_filename, delimiter):
    """
    Parses a file with one line, with values separated with tabs or commas.

    Args:
        oneline_filename: a file with one line
        delimiter (string): the kind of delimiter with values "," or "\t"

    Returns:
        maximums: a list
    """
    maximums = list()
    maximums_file = open(oneline_filename,'r')
    for line in maximums_file:
        word = line.split(delimiter)
        for w in word:
           maximums.append(w.rstrip())
    return maximums

def transform_labels_to_numeric(labels, unique_labels):
    """
    Transforms alphanumeric labels to numeric ones in the case of two classes.

    Args:
        labels: the input labels
        unique_labels: the list with the two unique labels

    Returns:
        new_labels: the encoded 0-1 new labels
    """
    labels_dict = {}
    labels_dict[unique_labels[0]] = 0
    labels_dict[unique_labels[1]] = 1
    logging.info(str(labels_dict))
    new_labels = [labels_dict[x] for x in labels]
    logging.info("Labels transformed to numeric successfully.")
    return new_labels

def transform_labels_to_alpha(labels, unique_labels):
    """
    Transforms numeric labels back to alphanumeric according to given unique_labels one to one mapping.

    Args:
        labels: the input labels
        unique_labels: the list with the two unique labels

    Returns:
        new_labels: the decoded labels with alphanumeric names
    """
    new_labels = []
    for x in labels:
        if x == 0:
            new_labels.append(unique_labels[0])
        elif x == 1:
            new_labels.append(unique_labels[1])
    logging.info("Labels transformed to alphanumeric successfully.")
    return new_labels

def normalize_testDataset(testSet, minimums, maximums, normalization_method):
    """
    Normalizes the test dataset according to one of the given methods

    Args:
        testSet: the input testset
        minimums (list): a list with the minimum values of each feature
        maximums (list): a list with the maximum values of each feature
        normalization_method: 1 for arithmetic sample-wise normalization and 2 for logarithmic normalization

    Returns:
        testSet: normalized data with the first method
        or
        logged_data: normalized data with the second method

    """
    if normalization_method == 1:
        # Arithmetic sample-wise normalization        
        for i in range(len(testSet)):
            for j in range((len(testSet[0]))):
                if (testSet[i][j]!='' and testSet[i][j]!=-1000):                    
                    if maximums[j] - minimums[j] == 0:
                        testSet[i][j] = testSet[i][j]
                    else:
                        testSet[i][j]=0+(float(testSet[i][j])- minimums[j] )/float(maximums[j]- minimums[j])        
        logging.info("Arithmetic sample-wise normalization successfully completed.")
        return testSet
    elif normalization_method == 2:
        # Logarithmic normalization
        logged_data = list()
        for i in range(len(testSet)):
            logged_data.append([])
            for j in range(len(testSet[0])):
                if testSet[i][j] == '' or testSet[i][j] == -1000:
                    logged_data[i].append('')
                else:
                    if(testSet[i][j] == 0):
                        logged_data[i].append(0)
                    else:
                        logged_data[i].append(math.log2(testSet[i][j]))
        logging.info("Logarithmic normalization successfully completed.")
        return logged_data


def perform_missing_value_imputation(dataset_initial, averages, missing_imputation_method=2):
    """
    Performs the missing value imputation.

    Args:
        dataset_initial (list of lists): the initial dataset
        averages (list): the average values for each sample
        missing_imputation_method: 1 for average imputation and 2 for KNN impute

    Returns:
        dataset_initial: the imputed data as a list of lists, using the first method
        or
        dataset: the imputed data as a list of lists, using the second method
    """
    if missing_imputation_method == 1:
        # Average imputation
        for i in range(len(dataset_initial)):
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] == -1000 or dataset_initial[i][j] == '':
                    dataset_initial[i][j] = averages[i]
        logging.info("Average  imputation successfully completed.")
        return dataset_initial
    else:
        # KNN impute
        dataset_initial=list(map(list, zip(*dataset_initial)))
        for i in range(len(dataset_initial)):
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                    dataset_initial[i][j] = np.NaN
        dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)), k=3)
        dataset = list(map(list, zip(*dataset)))
        logging.info("KNN imputation successfully completed.")
        return dataset

def random_bits():
    x = random.random()
    if x < 0.5:
        return 0
    else:
        return 1

def random_binary_class(unique_labels):
    x = random.random()
    if x < 0.5:
        return unique_labels[0]
    else:
        return unique_labels[1]

def geometric_mean_normalization(dataset_imputed, selected_commorbidities, output_message):
    """
    It does a normalization of the data based on the geometric mean.

    Args:
        dataset_imputed (list of lists): the initial dataset
        selected_commorbidities (list): the names (or lines) of the selected genes that with which the geometric mean will be calculated
        output_message: the final output message

    Returns:
        dataset_normalized:
        output_message:
    """
    # print(selected_commorbidities)
    positions_of_coms = list()
    geometric_means = list()
    geo_mean = 1
    dataset_imputed = list(map(list, zip(*dataset_imputed))) # (rows, cols): (features, samples) -> (samples, features)
    for i in range(len(dataset_imputed)):   # samples
        #geo_mean = 1
        for j in range(len(dataset_imputed[0])): # features
            if j in selected_commorbidities:
                if float(dataset_imputed[i][j]) == 0.0:
                    raise ValueError("Error in geometric normalization. Gene {} contains zero values. Choose different gene.".format(j))
                else:
                    geo_mean *= float(dataset_imputed[i][j])
        geo_mean = (geo_mean)**(1./len(selected_commorbidities))
        geometric_means.append(geo_mean)    # different geo_mean per sample
        geo_mean = 1
    for i in range(len(dataset_imputed)):
        for j in range(len(dataset_imputed[0])):
            if (dataset_imputed[i][j] != '' and dataset_imputed[i][j] != -1000):
                dataset_imputed[i][j] = float(dataset_imputed[i][j])/geometric_means[i]
    dataset_normalized = list(map(list, zip(*dataset_imputed))) # (samples, features) -> (features, samples)
    logging.info("Geometric mean normalization was used !\n")
    output_message += 'Geometric mean normalization was used!\n'
    return [dataset_normalized, output_message]

def parse_selected_features_string(astring):
    """
    Parses a string and strips it from commas or newline characters.

    Args:
        astring: the input string with comma separated or newline separated values

    Returns:
        A list with the substrings of the original string.
    """
    if "," in astring:
        return astring.split(",")
    elif "\\n" in astring:
        return astring.split("\\n")
    else:
        return astring

def parse_data(data_filename, delimiter):
    """
    Parses data.

    Args:
        data_filename: dataset filename
        delimiter (string): the kind of delimiter with values "," or "\t"

    Returns: a list of three lists, [proteins, data, samples].
    """
    num_of_lines = 0
    proteins = list()
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            if num_of_lines == 0:
                for j in range(len(line)):
                    if j > 0:
                        samples.append(line[j].strip())
            else:
                proteins.append(line[0])
                data.append([])
                for j in range(len(line)):
                    if j > 0:
                        if line[j] != '' and line[j] != "#VALUE!":
                            data[num_of_lines-1].append(float(line[j]))
                        else:
                            data[num_of_lines-1].append('')
            num_of_lines += 1
    logging.info('Data were successfully parsed!')
    return [proteins,data,samples]

def new_parse_data(data_filename, delimiter):
    """
    Parses data.

    Args:
        data_filename: dataset filename
        delimiter (string): the kind of delimiter with values "," or "\t"

    Returns: a list of three lists, [proteins, data].
    """
    num_of_lines = 0
    proteins = list()
    data = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            proteins.append(line[0])
            data.append([])
            for j in range(len(line)):
                if j > 0:
                    if line[j] != '' and line[j] != "#VALUE!":
                        data[num_of_lines].append(float(line[j]))
                    else:
                        data[num_of_lines].append('')
            num_of_lines += 1
    logging.info('Data were successfully parsed!')
    return [proteins,data]

def parse_data_with_only_samples(data_filename, delimiter):
    """
    Parses data.

    Args:
        data_filename: dataset filename with only data and samples
        delimiter (string): the kind of delimiter with values "," or "\t"

    Returns: a list of two lists, [data, samples].
    """
    num_of_lines = 0
    proteins = list()
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            if num_of_lines == 0:
                for j in range(len(line)):
                    samples.append(line[j].strip())
            else:
                proteins.append(line[0])
                data.append([])
                for j in range(len(line)):
                    try:
                        data[num_of_lines-1].append(float(line[j]))
                    except:
                        data[num_of_lines-1].append('')
            num_of_lines += 1
    #print('Data were successfully parsed!')
    return [data,samples]

def parse_only_dataset(dataset_filename, delimiter):
    """
    Parses a dataset which has no headers at all.

    Args:
        dataset_filename (string): the dataset filename
        delimiter (string): the kind of delimiter with values "," or "\t"

    Returns:
        data (list): a list of lists with the data

    """
    data = list()
    num_of_lines = 0
    with open(dataset_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            data.append([])
            for j in range(len(line)):
                if line[j] != '':
                    data[num_of_lines].append(float(line[j]))
                else:
                    data[num_of_lines].append('')
            num_of_lines += 1
    logging.info('Data were successfully parsed!')
    return data

def create_feature_list(dataset):
    """
    Creates a feature list with dummy names for a given dataset.

    Args:
        dataset (list): a list of lists

    Returns:
        (list): a one dimensional list with strings "Feature_0", "Feature_1", etc.
    """
    n = len(dataset)
    return ["Feature_" + str(i) for i in range(n)]

def create_samples_list(dataset):
    """
    Creates a samples list with dummy names for a given dataset.

    Args:
        dataset (list): a list of lists

    Returns:
        (list): a one dimensional list with strings "Sample_0", "Sample_1", etc.
    """
    n = len(dataset[0])
    return ["Sample_" + str(i) for i in range(n)]

def print_data(list_of_lists, output_filename):
    """
    Prints the data of a list of lists to a file.

    Args:
        list_of_lists (list): the list of lists
        output_filename (string): the output filename
    """
    with open(output_filename, "w") as handle:
        for i in range(len(list_of_lists)):
            for j in range(len(list_of_lists[0])):
                handle.write(str(list_of_lists[i][j]))
                if j < len(list_of_lists) - 1:
                    handle.write("\t")
            handle.write("\n")

def print_data_with_markers_and_samples(data, markers, samples, folder_name, filename):
    """
    Writes data, markers and samples to a file.

    Args:
        data: input data (list of lists)
        markers: input biomarkers (list)
        samples: input samples (list)
        folder_name: output folder
        filename: output filename

    Returns: doesn't return anything.
    """
    file = open(folder_name+filename,'w')
    message = ''
    for i in range(len(data[0])):
        message = message + '\t' + samples[i]
    message += '\n'
    for i in range(len(data)):
        message += markers[i]
        for j in range(len(data[0])):
            message += '\t' + str(data[i][j])
        message += '\n'
    file.write(message)
    file.close()

def average_duplicate_measurements(dataset_initial, markers):
    """
    Average duplicate measurements.

    Args:
        dataset_initial: the initial dataset, a list of lists
        markers: input biomarkers

    Returns: a list of two lists, data (a list of lists) and markers (a single list).
    """
    dataset = {}
    dict_of_occurences = {}
    num_of_elements = 0
    for i in range(len(dataset_initial)):
        if dataset_initial[i][0] not in dataset:
            dict_of_occurences[markers[i]] = 1
            dataset[markers[i]] = list()
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] != -1000 and dataset_initial[i][j] != '':
                    dataset[markers[i]].append(float(dataset_initial[i][j]))
                else:
                    dataset[markers[i]].append(0)
        else:
            dict_of_occurences[markers[i]] += 1
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] != -1000 and dataset_initial[i][j] != '':
                    dataset[markers[i]][j] = dataset[markers[i]][j] + float(dataset_initial[i][j])
        num_of_elements += 1
    element = 0
    for key in dataset:
        for j in range(len(dataset[key])):
            dataset[key][j] = dataset[key][j] / dict_of_occurences[key]
    data = list()
    markers = list()
    num_of_markers = 0
    for key,vals in dataset.items():
        data.append([])
        markers.append(key)
        for i in range(len(vals)):
            data[num_of_markers].append(vals[i])
        num_of_markers += 1
    logging.info("Averaging duplicate measurements completed successfully!")
    return [data,markers]

def has_negatives(list_of_lists):
    """
    Checks if a list of lists contains negative numbers.

    Args:
        list_of_lists (list): the input list

    Returns:
        (boolean): True if yes, False if no

    """
    for x in list_of_lists:
        for a in x:
            if a < 0: return True
    return False

def has_negatives_single_list(alist):
    """
    Checks if a single list contains negative numners.

    Args:
        alist (list): the input list

    Returns:
        (boolean): True if yes, False if no
    """
    for x in alist:
        if x < 0: return True
    return False


def fix_negative_values(list_of_lists, variables_for_normalization_nums):
    """
    For every row denoted by the variables_for_normalization_nums we look for the biggest negative number (with biggest distance from zero)
    and add its absolute value to the elements of that row.

    Args:
        list_of_lists (list): the input list

    Returns:
        list_of_lists (list): the input list without negative values in the selected variables for normalization

    """
    for row in variables_for_normalization_nums:
        if has_negatives_single_list(list_of_lists[row]):
            minimum = min(list_of_lists[row])
            for col in range(len(list_of_lists[row])):
                list_of_lists[row][col] += abs(minimum)
    return list_of_lists

def find_delimiter(dataset_filename):
    """
    Figures out which delimiter is being used in given dataset.

    Args:
        dataset_filename (string): the dataset filename

    Returns:
        (string): "," if CSV content, "\t" if TSV content.
    """
    with open(dataset_filename, 'r') as handle:
        head = next(handle)
    if "\t" in head: 
        return "\t"
    elif "," in head:
        return ","
    elif "," and "\t" in head: # The case where the comma is the decimal separator (greek system)
        return "\t"

def find_delimiter_labels(dataset_filename):
    """
    Figures out which delimiter is being used in given labels dataset.

    Args:
        dataset_filename (string): the dataset filename

    Returns:
        (string): "," if CSV content, "\t" if TSV content.
    """
    with open(dataset_filename, 'r') as handle:
        head = next(handle)
    if "\t" in head:
        return "\t"
    elif "," in head:
        return ","
    elif "," and "\t" in head: # The case where the comma is the decimal separator (greek system)
        return "\t"

def predictor(testset_filename, testset_labels_filename, maximums_filename, minimums_filename, averages_filename, features_filename,
                missing_imputation_method, normalization_method, output_folder, model_filename, data_been_preprocessed_flag, variables_for_normalization_string,
              filetype, has_features_header, has_samples_header, training_labels_filename, length_of_features_filename,
              length_of_features_from_training_filename, selected_comorbidities_string):
    """
    Predicts the labels of a given testset and test labels, according to a trained model.

    Args:
        testset_filename: (String) the file of the preprocessed testset
        testset_labels_filename: (String) the filename of the labels of the testSet
        maximums_filename: (String) the filename with the maximum values of each feature
        minimums_filename: (String) the filename with the minimum values of each feature
        averages_filename: (String) the filename with the average values for each sample
        features_filename: (String) the filename with the indexes of the selected features extracted from the training step
        normalization_method: (Integer) 1 for arithmetic, 2 for logarithmic
        missing_imputation_method: (Integer) 1 for average imputation, 2 for KNN imputation
        output_folder: (String) the name of the output folder
        model_filename: (String) the model filename
        data_been_preprocessed_flag: (Integer) 0 if data haven't been preprocessed and 1 if they have
        variables_for_normalization_string: (String) the selected variables for normalization as a string with comma or newline separated strings, eg. "ABC,DEF,GHI"
        filetype: 7 if it is a file from bionets with only features, not 7 if it's  not a file from bionets
        has_features_header: 1 if it has features header, 0 if it doesn't have
        has_samples_header: 1 if it has samples header, 0 if it doesn't have
        training_labels_filename: (String) the filename of the training set labels
        length_of_features_filename: (String) the filename with the length of features of the training set (from step 02)
        length_of_features_from_training_filename: (String) the filename with the length of features of the training set (from step 04)
        selected_comorbidities_string: (String) the selected comorbidities in the step 04 that have been deleted from the original dataset

    Returns:
        Nothing

    Prints to file:
        result_labels_output: the final labels encoded in the original encoding.
    """

    try:
        # Parsing selected features indexes
        if features_filename:
            features = parsing_oneline(features_filename)
            features = [int(x) for x in features]
        else:
            raise ValueError("Please provide a selected_features filename !")
        variables_for_normalization = parse_selected_features_string(variables_for_normalization_string)

        # Find delimiter for input dataset
        delim = find_delimiter(testset_filename)

        # Parsing test set
        if filetype != 7:
            if has_features_header and has_samples_header:
                markers, testSet, samples = parse_data(testset_filename, delim)

            elif has_features_header and not has_samples_header:
                markers, testSet = new_parse_data(testset_filename, delim)
                samples = create_samples_list(testSet)

            elif not has_features_header and has_samples_header:
                testSet, samples = parse_data_with_only_samples(testset_filename, delim)
                markers = create_feature_list(testSet)

            else: # has nothing
                testSet = parse_only_dataset(testset_filename, delim)
                markers = create_feature_list(testSet)
                samples = create_samples_list(testSet)
        else:
            if has_features_header and has_samples_header:
                markers, testSet, samples = parse_data(testset_filename, delim)

            elif has_features_header and not has_samples_header:
                markers, testSet = new_parse_data(testset_filename, delim)
                samples = create_samples_list(testSet)

            elif not has_features_header and has_samples_header:
                testSet, samples = parse_data_with_only_samples(testset_filename, delim)
                markers = create_feature_list(testSet)

            else: # has nothing
                testSet = parse_only_dataset(testset_filename, delim)
                markers = create_feature_list(testSet)
                samples = create_samples_list(testSet)

        old_testSet = copy.deepcopy(testSet)

        # If the user doesn't give the testSet with the selected features only, but instead gives the full dataset
        if length_of_features_filename:
            length_of_features = parsing_oneline(length_of_features_filename)
            list_of_deleted_features_in_step2 = [int(x) for x in length_of_features[1:]]
            length_of_features = int(length_of_features[0])
            selected_comorbidities = parse_selected_features_string(selected_comorbidities_string)

            if len(testSet) == length_of_features-len(selected_comorbidities)-len(list_of_deleted_features_in_step2) or len(testSet) == length_of_features:  # total number of features in the training set
                testSet = [testSet[i] for i in range(len(testSet)) if i not in list_of_deleted_features_in_step2]
                indexes_of_testSet_lines = [i for i in range(len(testSet)) if i not in list_of_deleted_features_in_step2]            
                
                if len(selected_comorbidities) != 0:                
                    testSet = np.transpose(testSet)
                    new_data = [[] for _ in range(len(testSet))]
                    for i in range(len(testSet)):
                        for j in range(len(testSet[0])):
                            if not markers[j] in selected_comorbidities:
                                new_data[i].append(testSet[i][j])
                    new_data = np.transpose(new_data)

                    del_com_testSet = new_data
                    testSet = del_com_testSet

                new_testSet = list()    
                for i in indexes_of_testSet_lines:
                    if i in features:
                        new_testSet.append(testSet[i])
                #new_testSet = [testSet[i] for i in features]
            elif len(testSet) == len(features): # total number of selected features            
                new_testSet = testSet
            else:
                raise ValueError("You have wrong number of features !")
        else:
            # If the user's training set hasn't been preprocessed in InSyBio's platform
            length_of_features_from_training = parsing_oneline(length_of_features_from_training_filename)
            length_of_features_from_training = int(length_of_features_from_training[0])
            if len(testSet) == length_of_features_from_training:
                new_testSet = [testSet[i] for i in features]
            elif len(testSet) == len(features):
                new_testSet = testSet
            else:
                raise ValueError("You have wrong number of features !")

        # Parsing testSet labels
        if testset_labels_filename:
            delim_labels = find_delimiter_labels(testset_labels_filename)
            testSet_labels = parsing_oneline_generic(testset_labels_filename, delim_labels)
            #testSet_labels.pop() # last element is a string '' (why?)
            unique_labels = list(set(testSet_labels))
        else:
            # if testset labels are not provided
            delim_tr_labels = find_delimiter_labels(training_labels_filename)
            training_labels = parsing_oneline_generic(training_labels_filename, delim_tr_labels)
            unique_labels = list(set(training_labels))
            testSet_labels = [random_binary_class(unique_labels) for _ in range(len(testSet[0]))]
    except Exception as e:
        logging.exception("Exception during parsing!")
        return [0, "Exception during parsing: {}".format(e)]

    # If the labels are alphanumeric encode them into numeric
    try:
        float(unique_labels[0])
        labels = testSet_labels
    except Exception as e:
        logging.info("Encoding alphanumeric labels into numeric.")
        labels = transform_labels_to_numeric(testSet_labels, unique_labels)

    try:
        if data_been_preprocessed_flag == 0:
            # Missing values imputation of initial dataset
            if features_filename and missing_imputation_method == 1:
                if averages_filename:
                    averages = parsing_oneline(averages_filename)
                    averages = [float(x) for x in averages]
                    newAverages = [averages[i] for i in features]
                    message = " "
                else:
                    newAverages = list()
                    message = "No missing imputation took place."
                dataset_mv_imputed = perform_missing_value_imputation(new_testSet, newAverages, missing_imputation_method)
            elif features_filename and missing_imputation_method == 2:
                newAverages = list()
                message = " "
                dataset_mv_imputed = perform_missing_value_imputation(new_testSet, newAverages, missing_imputation_method)
            else:
                raise ValueError("Please provide a selected_features filename !")

            # Geometric normalization
            if variables_for_normalization:
                # Catch the cases where the user inputs only one element with a comma/newline at the end
                if '' in variables_for_normalization and not isinstance(variables_for_normalization, str):
                    variables_for_normalization.pop()

                markers = [markers[i] for i in range(len(markers)) if i not in list_of_deleted_features_in_step2]
                new_markers = list()
                for i in indexes_of_testSet_lines:
                    if i in features:
                        try:
                            new_markers.append(markers[i])
                        except:
                            new_markers.append("MarkerNotFound")

                if isinstance(variables_for_normalization, list):
                    variables_for_normalization_nums = list()
                    for variable in variables_for_normalization:
                        try:
                            variable_index = new_markers.index(variable)
                            variables_for_normalization_nums.append(variable_index)
                        except Exception as e:
                            logging.exception("The biomarker(s) provided are not in the list of the biomarkers of the input file !")
                            pass
                elif isinstance(variables_for_normalization, str):
                    variables_for_normalization_nums = list()
                    try:
                        variable_index = new_markers.index(variables_for_normalization)
                        variables_for_normalization_nums.append(variable_index)
                    except Exception as e:
                        logging.exception("The biomarker(s) provided are not in the list of the biomarkers of the input file !")
                        pass

                output_message = ""
                if variables_for_normalization_nums:
                    dataset_mv_imputed = fix_negative_values(dataset_mv_imputed, variables_for_normalization_nums)
                    [dataset_mv_imputed, output_message] = geometric_mean_normalization(dataset_mv_imputed, variables_for_normalization_nums, output_message)
                    #print_data_with_markers_and_samples(dataset_imputed, new_markers, samples, output_folder, "dataset_normalized_mvimputed_geo.txt") 

            # Normalization of initial test dataset
            if features_filename:
                if normalization_method == 1:
                    if maximums_filename and minimums_filename:
                        maximums = parsing_oneline(maximums_filename)
                        maximums = [float(x) for x in maximums]
                        minimums = parsing_oneline(minimums_filename)
                        minimums = [float(x) for x in minimums]
                        features = [int(x) for x in features]
                        newMinimums = [minimums[i] for i in features]   # IS THIS DONE RIGHT ? FILTERING THE RIGHT INDEXES ?
                        newMaximums = [maximums[i] for i in features]   # IS THIS DONE RIGHT ? FILTERING THE RIGHT INDEXES ?
                        message = " "
                        dataset_mv_imputed = np.transpose(dataset_mv_imputed)
                        normalized_mv_imputed_dataset = normalize_testDataset(dataset_mv_imputed, newMinimums, newMaximums, normalization_method)
                        normalized_mv_imputed_dataset = np.transpose(normalized_mv_imputed_dataset)
                        inputs = normalized_mv_imputed_dataset
                    else:                        
                        message += " No normalization took place."
                        inputs = new_testSet
                else:
                    maximums = list()
                    minimums = list()
                    newMinimums = list()
                    newMaximums = list()
                    message = " "
                    if has_negatives(dataset_mv_imputed):
                        raise ValueError("Your data contain negative values. Logarithmic normalization is not supported.")
                    dataset_mv_imputed = np.transpose(dataset_mv_imputed)
                    normalized_mv_imputed_dataset = normalize_testDataset(dataset_mv_imputed, newMinimums, newMaximums, normalization_method)
                    normalized_mv_imputed_dataset = np.transpose(normalized_mv_imputed_dataset)
                    inputs = normalized_mv_imputed_dataset
                                
            else:
                raise ValueError("Please provide a selected_features filename !")
                       
        else:
            message = " "
            inputs = new_testSet
    except Exception as e:
        logging.exception("Exception during preprocessing!")
        return [0, "Exception during preprocessing: {}".format(e)]


    print(inputs[0])
    # Preparing for modelling -----------------
    try:
        # Inputs must be the selected subset of testSet, always !!! Or else svm_predict won't work correctly
        outputs = labels

        testing_inputs = inputs
        testing_outputs = outputs
        testing_outputs = [int(x) for x in testing_outputs]
        logging.info("testing outputs:" + str(testing_outputs))

        # Loading the model
        model = svm_load_model(model_filename)

        testing_inputs = list(map(list, zip(*testing_inputs)))
        [p_labs, p_acc, dec_values] = svm_predict(testing_outputs, testing_inputs, model)

        # if true labels are provided, calculate accuracy, sensitivity and specificity
        if testset_labels_filename:
            temp_labels = testing_outputs
            accuracy=0
            sensitivity=0
            geometric_mean=0
            specificity=0
            accuracy=accuracy+p_acc[0]
            tp=0
            fp=0
            tn=0
            fn=0
            for i in range(len(temp_labels)):
                if float(temp_labels[i]) == float(p_labs[i]) and float(temp_labels[i]) == 1.0:
                    tp+=1
                elif float(temp_labels[i]) == float(p_labs[i]) and float(temp_labels[i]) == 0.0:
                    tn+=1
                elif float(temp_labels[i]) != float(p_labs[i]) and float(temp_labels[i]) == 0.0:
                    fp+=1
                else:
                    fn+=1
            logging.info('tp='+str(tp))
            print('tp='+str(tp))
            logging.info('tn='+str(tn))
            print('tn='+str(tn))
            logging.info('fp='+str(fp))
            print('fp='+str(fp))
            logging.info('fn='+str(fn))
            print('fn='+str(fn))
            if tp + fn != 0:
                s_sensitivity = tp/float(tp+fn)
            else:
                s_sensitivity = 1

            if tn + fp != 0:
                s_specificity = tn/float(tn+fp)
            else:
                s_specificity = 1
            sensitivity += s_sensitivity
            specificity += s_specificity
            logging.info('Test set accuracy: '+str(accuracy))
            logging.info('Test set sensitivity: '+str(sensitivity))
            logging.info('Test set specificity: '+str(specificity))
            if tp+fp != 0:
                precision = tp/float(tp+fp)

            if tp+fp != 0 and precision+sensitivity !=0:
                F1 = 2*(precision*sensitivity)/(precision+sensitivity)
                print("F1=",F1)

            if tp+fp != 0 and tp+fn!=0 and tn+fp !=0 and tn+fn != 0:
                MCC = (tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                print("MCC=",MCC)
            accuracy = p_acc[0]
            logging.info('Test set accuracy: '+str(accuracy))
            sensitivity = 100*sensitivity
            specificity = 100*specificity
            geometric_mean = math.sqrt(sensitivity*specificity)            
            with open(output_folder + "metrics.txt","w") as metrics_handle:
                metrics_handle.write("Test set accuracy: " + "{0:.2f}".format(accuracy) + "%" + "\n")
                metrics_handle.write("Test set sensitivity: " + "{0:.2f}".format(sensitivity) + "%" + "\n")
                metrics_handle.write("Test set specificity: " + "{0:.2f}".format(specificity) + "%" + "\n")
                metrics_handle.write("Test set geometric mean: " + "{0:.2f}".format(geometric_mean) + "%" + "\n")


        # Saving result labels
        result_labels_output = copy.deepcopy(p_labs)
        result_labels_output = list(map(int, result_labels_output))

        # If the original labels where alphanumeric then decode the labels to that format
        try:
            float(unique_labels[0])
        except Exception as e:
            logging.info("Decoding the labels back to alphanumeric format.")
            result_labels_output = transform_labels_to_alpha(result_labels_output, unique_labels)
        logging.info(str(result_labels_output))

        result_labels_fid = open(output_folder + "result_labels"  + ".txt", "w")
        for label in result_labels_output:
            result_labels_fid.write(str(label) + "\t")
        result_labels_fid.close()
    except Exception as e:
        logging.exception("Exception during modelling!")
        return [0, "Exception during modelling: {}".format(e)]
    logging.info("Successfully finished!")
    
    message += " Successful completion !"
    if (missing_imputation_method == 1 and not averages_filename) or (normalization_method == 1 and not minimums_filename):
        return [2, message]
    return [1, "Successful completion !"]

if __name__ == "__main__":
    testset_filename = sys.argv[1]
    testset_labels_filename = sys.argv[2]
    maximums_filename = sys.argv[3]
    minimums_filename = sys.argv[4]
    averages_filename = sys.argv[5]
    features_filename = sys.argv[6]
    missing_imputation_method = int(sys.argv[7])
    normalization_method = int(sys.argv[8])
    model_filename = sys.argv[9]
    data_been_preprocessed_flag = int(sys.argv[10])
    variables_for_normalization_string = sys.argv[11]
    filetype = int(sys.argv[12])
    has_features_header = int(sys.argv[13])
    has_samples_header = int(sys.argv[14])
    training_labels_filename = sys.argv[15]
    length_of_features_filename = sys.argv[16]
    selected_comorbidities_string = sys.argv[17]

    tstamp = time.strftime('%Y_%m_%d')
    output_folder = 'predictor_results/' + str(tstamp) + '_' + str(time.time()) + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    result = predictor(testset_filename, testset_labels_filename, maximums_filename, minimums_filename, averages_filename, features_filename,
                missing_imputation_method, normalization_method, output_folder, model_filename, data_been_preprocessed_flag, variables_for_normalization_string,
                filetype, has_features_header, has_samples_header, training_labels_filename, length_of_features_filename, selected_comorbidities_string)
    print(result)
