"""
This is the script for dataset preprocessing. It does normalization, filtering, missing value imputation
and outlier detection with PCA LOF method of the input dataset.

Example run:

python3 dataset_preprocessing.py Input/example_dataset.txt "ACTB,VIME,APOE,TLN1,CO6A3" Output/
output_dataset_strings.txt 0.1 1 2 8 1 1
python3 dataset_preprocessing.py Input/example_dataset.txt "ACTB\nVIME\nAPOE\nTLN1\nCO6A3" Output/
output_dataset_strings.txt 0.1 1 2 8 1 1

python3 dataset_preprocessing.py Input/Multiple/example_dataset_full4.txt.csv "YWHAZ,SERPINA1,SERPINF2,AGT,ANTXR1"
Output/ output_dataset_strings_multi.txt 0.1 1 2 8 1 1


"""
import os
import sys
import csv
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)
import datetime
import logging
import math
import numpy as np
from math import sqrt
from sklearn import preprocessing
import pandas as pd


def parse_data(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting both feature and sample headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of three lists, [features, data, samples].
    """
    num_of_lines = 0
    features = list()
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            if num_of_lines == 0:
                for j, value in enumerate(line):
                    if j > 0:
                        samples.append(value.strip())
            else:
                data.append([])
                for j, value in enumerate(line):
                    if j == 0:
                        features.append(value.strip())
                    else:
                        if value != '' and value != "#VALUE!":
                            data[num_of_lines - 1].append(float(value))
                        else:
                            data[num_of_lines - 1].append(-1000)
            num_of_lines += 1
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [features, data, samples]


def new_parse_data(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting only feature headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of two lists, [features, data].
    """
    num_of_lines = 0
    features = list()
    data = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            data.append([])
            for j, value in enumerate(line):
                if j == 0:
                    features.append(value)
                else:
                    if value != '' and value != "#VALUE!":
                        data[num_of_lines].append(float(value))
                    else:
                        data[num_of_lines].append(-1000)
            num_of_lines += 1
            # print('Data were successfully parsed!')
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [features, data]


def parse_data_with_only_samples(data_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting only sample headers
    :param data_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list of two lists, [data, samples].
    """
    num_of_lines = 0
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            for value in line:
                if num_of_lines == 0:
                    samples.append(value.strip())
                else:
                    data.append([])
                    if value != '' and value != "#VALUE!":
                        data[num_of_lines - 1].append(float(line[j]))
                    else:
                        data[num_of_lines - 1].append(-1000)
                num_of_lines += 1
    # print('Data were successfully parsed!')
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return [data, samples]


def parse_only_dataset(dataset_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses dataset and splits it into Features, sample_name and data lists, expecting no headers
    :param dataset_filename: dataset filename
    :param delimiter: the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: a list data, data.
    """
    data = list()
    num_of_lines = 0
    with open(dataset_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter=delimiter):
            data.append([])
            for value in line:
                if value != '' and value != '#VALUE!':
                    data[num_of_lines].append(float(value))
                else:
                    data[num_of_lines].append('')
            num_of_lines += 1
    # logging.info('Data were successfully parsed!')
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return data


def parse_selected_features(features_filename, delimiter, user='unknown', jobid=0, pid=0):
    """
    Parses the selected features filename.

    Args:
        features_filename (string): the selected features filename, one line tab seperated values
        delimiter (string): the kind of delimiter with values "," or "\t"
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns:
        features (list): the list of the selected features
    """
    features = list()
    try:
        with open(features_filename) as features_fname:
            for line in csv.reader(features_fname, delimiter=delimiter):
                for i in range(len(line)):
                    features.append(line[i].strip())  # returns a list of one string eg. ['1 2 3']
        features = list(map(int, features[0].split()))  # returns a list of ints eg. [1,2,3]
        logging.info('PID:{}\tJOB:{}\tUSER:{}\tFeatures were successfully parsed!'.format(pid, jobid, user))
        return features
    except Exception:
        logging.exception('PID:{}\tJOB:{}\tUSER:{}\tEmpty selected features file provided!'.format(pid, jobid, user))


def find_delimiter(dataset_filename):
    """
    Figures out which delimiter is being used in given dataset.

    Args:
        dataset_filename (string): the dataset filename

    Returns:
        (string): "," if CSV content, "\t" if TSV content.
    """
    with open(dataset_filename, 'r') as handle:
        # head = next(handle)
        head = next(handle)
    if "\t" in head:
        return "\t"
    elif "," in head:
        return ","
    elif "," and "\t" in head:  # The case where the comma is the decimal separator (greek system)
        return "\t"


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


def write_one_dimensional_list_to_tab_delimited_file(data, filename):
    """
    Writes one dimensional list to tab delimited file.

    Args:
        data: input data
        filename: output filename

    Returns: doesn't return anything, only writes data to file.
    """
    with open(filename, 'w') as file_id:
        for i in range(len(data)):
            file_id.write(str(data[i]))
            file_id.write('\n')


def outlier_detection(dataset, folder_name):
    """
    Detects the outliers.

    Args:
        dataset: input dataset
        folder_name: output folder name

    Returns: a list of lists
    """
    pca = PCA(n_components=0.9, svd_solver='full')
    new_dataset = list()
    num_of_samples = 0
    for j in range(len(dataset[0])):
        new_dataset.append([])
        for i in range(len(dataset)):
            new_dataset[num_of_samples].append(float(dataset[i][j]))
        num_of_samples += 1
    dataset_new = pca.fit_transform(new_dataset)
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(dataset_new)
    return dataset_new


def filter_proteomics_dataset(dataset_initial, proteins, percentage, output_message, output_folder_name, user='unknown',
                              jobid=0, pid=0):
    """
    Filters proteomics dataset.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        proteins: a list of proteins
        percentage: a float
        output_message: the output message
        output_folder_name: path to folder for output files
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: a list with new_data (filtered data list), new_proteins (filtered proteins), output message (string).
    """
    new_data = list()
    selected = 0
    new_proteins = list()
    missing_proteins = 0
    missing_proteins_list = list()
    proteins_missing_values_percentage = 0
    for i in range(len(dataset_initial)):
        missing = 0
        for j in range(len(dataset_initial[0])):
            if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                missing += 1
                proteins_missing_values_percentage += 1

        if len(dataset_initial[0]) != 0 and missing / float(len(dataset_initial[0])) < percentage:
            # if missing/float(len(dataset_initial[0])) < percentage:
            # print(i)
            new_data.append([])
            for k in range(len(dataset_initial[i])):
                new_data[selected].append(dataset_initial[i][k])
            selected += 1
            new_proteins.append(proteins[i])
        else:
            missing_proteins_list.append(i)
            missing_proteins += 1

    with open(output_folder_name + "length_of_features.txt", "a") as handle:
        handle.write("\n")
        for i in missing_proteins_list:
            handle.write(str(i))
            handle.write("\n")
    try:
        output_message += "Data were successfully filtered!\nResults of filtering:\n"
        output_message += "Total Number of Molecules = {}\n".format(str(len(dataset_initial)))
        output_message += "Total Number of Molecules with missing values less than the allowed threshold = {}\n".format(
            str(selected))
        percentage_missing_values = proteins_missing_values_percentage / float(
            len(dataset_initial) * len(dataset_initial[0]))
        output_message += "Percentage of Missing Values in all molecules = {0:.2f}\n".format(percentage_missing_values)
        logging.info('PID:{}\tJOB:{}\tUSER:{}\t{}'.format(pid, jobid, user, output_message))
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError during filtering".format(pid, jobid, user))
        raise ValueError('Error during filtering', e)
    return [new_data, new_proteins, output_message]


def perform_missing_value_imputation(dataset_initial, averages, missing_imputation_method, user='unknown', jobid=0,
                                     pid=0):
    """
    Perform missing values imputation.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        missing_imputation_method (integer): 1 for average imputation, 2 for KNN-impute
        averages (list): list of averages per feature, used for imputation
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: a list with the final dataset (list of lists) and the output message (string).
    """
    # missing values imputation
    if missing_imputation_method == 1:
        # average imputation
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tAverage imputation method was used!".format(pid, jobid, user))
        for i in range(len(dataset_initial)):
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] == -1000 or dataset_initial[i][j] == '':
                    dataset_initial[i][j] = averages[i]
        return dataset_initial
    else:
        # KNN-impute
        dataset_initial = list(map(list, zip(*dataset_initial)))
        for i in range(len(dataset_initial)):
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                    dataset_initial[i][j] = np.NaN
        dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)), k=3)
        dataset = list(map(list, zip(*dataset)))
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tKNN imputation method was used!".format(pid, jobid, user))
        return dataset


def normalize_dataset(dataset_initial, minimums, maximums, normalization_method, user='unknown', jobid=0, pid=0):
    """
    Normalize the Test dataset, according to Training parameters.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        minimums (list): a list with the minimum values of each feature
        maximums (list): a list with the maximum values of each feature
        normalization_method (integer): 1 for arithmetic sample-wise normalization, 2 for logarithmic normalization
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: if method 1 selected a list with the normalized dataset and the output message, else for method 2
            logged data are returned along with the output message.
    """
    if normalization_method == 1:
        # Arithmetic sample-wise normalization
        outdata_data = [[]]*len(dataset_initial)
        for i in range(len(dataset_initial)):
            outdata_data[i] = [0]*len(dataset_initial[0])
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] != '' and dataset_initial[i][j] != -1000:
                    if maximums[i] - minimums[i] == 0:
                        outdata_data[i][j] = dataset_initial[i][j]
                    else:
                        outdata_data[i][j] = 0 + (float(dataset_initial[i][j]) - minimums[i]) / float(
                            maximums[i] - minimums[i])
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tArithmetic normalization was used!".format(pid, jobid, user))
        return outdata_data
    else:
        # Logarithmic normalization
        # dataset_initial = np.transpose(dataset_initial)
        logged_data = list()
        for i in range(len(dataset_initial)):
            logged_data.append([])
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] == '' or dataset_initial[i][j] == -1000:
                    logged_data[i].append('')
                else:
                    if dataset_initial[i][j] == 0:
                        logged_data[i].append(0)
                    else:
                        logged_data[i].append(math.log2(dataset_initial[i][j]))

        logging.info("PID:{}\tJOB:{}\tUSER:{}\tLogarithmic normalization was used!".format(pid, jobid, user))
        # logged_data = np.transpose(logged_data)
        return logged_data


def geometric_mean_normalization(dataset_imputed, selected_features, output_message, user='unknown', jobid=0, pid=0):
    """
    It does a normalization of the data based on the geometric mean.

    Args:
        dataset_imputed (list of lists): the initial dataset
        selected_features (list): the names (or lines) of the selected genes that with which the geometric mean will
         be calculated
        output_message: the final output message
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns:
        dataset_normalized:
        output_message:
    """

    geometric_means = list()
    geo_mean = 1
    dataset_imputed = np.transpose(dataset_imputed)  # (rows, cols): (features, samples) -> (samples, features)
    for i in range(len(dataset_imputed)):  # samples
        for j in range(len(dataset_imputed[0])):  # features
            if j in selected_features:
                if float(dataset_imputed[i][j]) == 0.0:
                    raise ValueError("Error in geometric normalization. Gene {} contains zero values. Choose different"
                                     " gene.".format(j))
                else:
                    geo_mean *= float(dataset_imputed[i][j])
        geo_mean = geo_mean ** (1. / len(selected_features))
        geometric_means.append(geo_mean)  # different geo_mean per sample
        geo_mean = 1
    # print(geometric_means)
    for i in range(len(dataset_imputed)):
        for j in range(len(dataset_imputed[0])):
            if dataset_imputed[i][j] != '' and dataset_imputed[i][j] != -1000:
                # print(dataset_imputed[i][j])
                dataset_imputed[i][j] = float(dataset_imputed[i][j]) / geometric_means[i]
    dataset_normalized = np.transpose(dataset_imputed)  # (samples, features) -> (features, samples)
    # print("Geometric mean normalization was used !\n")
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tGeometric mean normalization was used !".format(pid, jobid, user))
    output_message += 'Geometric mean normalization was used!\n'
    return [dataset_normalized, output_message]


def average_duplicate_measurements(dataset_initial, markers, user='unknown', jobid=0, pid=0):
    """
    Average duplicate measurements.

    Args:
        dataset_initial: the initial dataset, a list of lists
        markers: input biomarkers
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: a list of two lists, data (a list of lists) and markers (a single list).
    """
    dataset = {}
    dict_of_occurences = {}
    num_of_elements = 0
    for i in range(len(dataset_initial)):
        if markers[i] not in dataset:
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

    for key in dataset:
        for j in range(len(dataset[key])):
            dataset[key][j] = dataset[key][j] / dict_of_occurences[key]
    data = list()
    markers = list()
    num_of_markers = 0
    for key, vals in dataset.items():
        data.append([])
        markers.append(key)
        for i in range(len(vals)):
            data[num_of_markers].append(vals[i])
        num_of_markers += 1
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tAveraging duplicate measurements completed successfully!".format(pid, jobid,
                                                                                                            user))
    return [data, markers]


def print_data(data, markers, labels, folder_name, filename):
    """
    Writes data and labels to a file.

    Args:
        data: input data (list of lists)
        markers: input biomarkers (list)
        labels: input labels (list)
        folder_name: output folder
        filename: output filename

    Returns: doesn't return anything, only writes labels and data to a file.
    """
    with open(folder_name + filename, 'w') as file:
        message = ''
        for i in range(len(data[0])):
            message = message + '\t' + labels[i]
        message += '\n'
        for i in range(len(data)):
            message += markers[i]
            for j in range(len(data[0])):
                message += '\t' + str(data[i][j])
            message += '\n'
        file.write(message)


def new_print_data(data, markers, folder_name, filename):
    """
    Writes data and labels to a file.

    Args:
        data: input data (list of lists)
        markers: input biomarkers (list)
        folder_name: output folder
        filename: output filename

    Returns: doesn't return anything, only writes labels and data to a file.
    """
    with open(folder_name + filename, 'w') as file:
        message = ''
        for i in range(len(data)):
            message += markers[i]
            for j in range(len(data[0])):
                message += '\t' + str(data[i][j])
            message += '\n'
        file.write(message)


def parse_selected_features_string(astring):
    """
    Parses a string and strips it from commas or newline characters.

    Args:
        astring: the input string with comma separated or newline separated values

    Returns:
        A list with the substrings of the original string.
    """
    # print(astring)
    if "," in astring:
        return astring.split(",")
    elif "\\n" in astring:
        return astring.split("\\n")
    else:
        # raise ValueError("The string doesn't contain comma separated values or newline separated values !")
        return astring


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
            if a < 0:
                return True
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
        if x < 0:
            return True
    return False


def fix_negative_values(list_of_lists, variables_for_normalization_nums):
    """
    For every row denoted by the variables_for_normalization_nums we look for the biggest negative number (with biggest
     distance from zero)
    and add its absolute value to the elements of that row.

    Args:
        list_of_lists (list): the input list
        variables_for_normalization_nums (index list): indexes of selected features to normalize

    Returns:
        list_of_lists (list): the input list without negative values in the selected variables for normalization

    """
    for row in variables_for_normalization_nums:
        if has_negatives_single_list(list_of_lists[row]):
            minimum = min(list_of_lists[row])
            for col in range(len(list_of_lists[row])):
                list_of_lists[row][col] += abs(minimum)
    return list_of_lists


def parsing_oneline(oneline_filename):
    """
    Parses a file with one line, with values separated with tabs. Also works with data separated with newlines for
    some reason.

    Args:
        oneline_filename: a file with one line
    Returns:
        maximums: a list
    """
    maximums = list()
    maximums_file = open(oneline_filename, 'r')
    for line in maximums_file:
        word = line.split("\t")
        for w in word:
            maximums.append(w.rstrip())
    return maximums


def preprocess_data(input_dataset, maximums, minimums, averages_filename, features_filename,
                    missing_imputation_method, normalization_method, output_folder_name, data_been_preprocessed_flag,
                    variables_for_normalization_string, has_features_header, has_samples_header, user='unknown',
                    jobid=0, pid=0):
    """
    A script that preprocesses the data. Parse the input files and create a list with the dataset and its features
    Rearrange the dataset in order to align the testing features to the training features order, also perform the
    preprocessing steps if preprocessing is selected

    Args:
        input_dataset: the initial dataset to be preprocessed
        variables_for_normalization_string: the string with the names of the selected genes that with which the
        geometric mean will be calculated, separated with commas or "\n"
        output_folder_name: the output folder name
        maximums: (list) a list with the maximum values of each feature
        minimums: (list) a list with the minimum values of each feature
        averages_filename: (String) the filename with the average values for each sample
        features_filename: (String) the filename with the indexes of the selected features extracted from the
        training step
        normalization_method: (Integer) 1 for arithmetic, 2 for logarithmic
        missing_imputation_method: (Integer) 1 for average imputation, 2 for KNN imputation
        data_been_preprocessed_flag: (Integer) 0 if data haven't been preprocessed and 1 if they have
        has_features_header: 1 if it has features header, 0 if it doesn't have
        has_samples_header: 1 if it has samples header, 0 if it doesn't have

        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns:
        output_dataset: the preprocessed dataset
    """

    try:
        # Find delimiter
        delim = find_delimiter(input_dataset)

        if has_features_header and has_samples_header:
            markers, testdata, samples = parse_data(input_dataset, delim, user, jobid, pid)

        elif has_features_header and not has_samples_header:
            markers, testdata = new_parse_data(input_dataset, delim, user, jobid, pid)
            samples = create_samples_list(testdata)

        elif not has_features_header and has_samples_header:
            testdata, samples = parse_data_with_only_samples(input_dataset, delim, user, jobid, pid)
            markers = create_feature_list(testdata)

        else:  # has nothing
            testdata = parse_only_dataset(input_dataset, delim, user, jobid, pid)
            markers = create_feature_list(testdata)
            samples = create_samples_list(testdata)

        variables_for_normalization = parse_selected_features_string(variables_for_normalization_string)

    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError during parsing Testing set for preprocessing.".format(
            pid, jobid, user))
        return [False, "Error during parsing the Dataset. Please contact info@insybio.com for more information", '']
    message = ''
    try:
        # Average duplicate measurements & outlier detection
        #missing_imputation_method = 0
        if missing_imputation_method != 0:
            [testdata, markers] = average_duplicate_measurements(testdata, markers, user, jobid, pid)
            message += "Duplicate measurements have been averaged successfully!\n"
            # print_data(dataset_imputed, new_features, samples, output_folder_name,
            # "dataset_averaged_duplicate_measurements.csv")
            pca_data = outlier_detection(testdata, output_folder_name)
            # data_transp=list(map(list, zip(*pca_data)))
        else:
            message += "Duplicate measurements have not been averaged!\n"
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during averaging.".format(
            pid, jobid, user))
        return [False, "Preprocessing raised the exception during averaging. Please contact info@insybio.com "
                       "for more information", '']

    try:
        features_training = []
        with open(features_filename, 'r') as features_file:
            for line in features_file:
                for feature in line.split(','):
                    features_training.append(str(feature).strip())
        # df = pd.read_csv(features_filename, header=None)
        # df_list = [item for sublist in df.values.tolist() for item in sublist]  # read flat list from dataframe
        # for i in range(len(df_list)):
        #     features_training.append(df_list[i])

        # Filter the Testset so it has the same features and in the same order as the Training set
        # def filter_features_predict(dataset_initial, features_training):
        new_data = []  # list to hold dataset after picking features present in training
        new_features = []  # initialise list to hold feature names after deletion
        column_count = len(testdata[0])  # counter for iterating on a feature across observations

        for i, feature in enumerate(features_training):  # match each feature in predict with training
            flag_found = 0  # flag to check if training feature was found in predict feature list
            for j, feat_values in enumerate(testdata):  # get the row index of predict data to be picked for matching
                # feature
                if feature == markers[j]:  # if feature names match
                    flag_found = 1
                    new_data.append(feat_values)  # list of list to hold dataset after matching features
                    new_features.append(markers[j])  # 1st value in list is feature
                    break

            # check if training features not in predict then add null value
            if flag_found == 0:
                logging.info("PID:{}\tJOB:{}\tUSER:{}\tTraining feature not found in predict. Adding null value "
                             "for {}".format(pid, jobid, user, features_training[i]))
                new_data.append([float("nan") for _ in range(column_count)])  # list of list to hold dataset after
                # matching features
                new_features.append(feature)  # training feature as not found in predict data

        logging.info("PID:{}\tJOB:{}\tUSER:{}\tFeatures successfully matched with training features".format(pid, jobid,
                                                                                                            user))
        # return [new_data, new_features]
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during feature selection!".format(pid, jobid, user))
        return [False, "Exception during feature selection", '']

    if data_been_preprocessed_flag == 0:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing is selected.".format(
                pid, jobid, user))
        """try:
            output_message = ""
            # Filter proteomics dataset
            [dataset_filtered, markers, output_message] = filter_proteomics_dataset(
                new_data, new_features, missing_threshold, output_message, output_folder_name)
        # print_data(dataset_filtered, markers, samples, output_folder_name, "data_mv_filtered.txt")
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during filtering.".format(
                pid, jobid, user))
            return [0, "Preprocessing raised an exception during Dataset filtering. Please contact info@insybio.com for"
                       " more information"]"""
        print('missing vaues imputation', missing_imputation_method)
        try:
            # Perform missing values imputation
            if missing_imputation_method == 1:
                if averages_filename:
                    averages = parsing_oneline(averages_filename)
                    averages = [float(x) for x in averages]
                    newAverages = [averages[i] for i in range(len(new_features))]

                    dataset_imputed = perform_missing_value_imputation(new_data, newAverages, missing_imputation_method,
                                                                       user, jobid, pid)
                else:
                    newAverages = list()
                    message += "No missing imputation took place.\n"
                    dataset_imputed = new_data
                # dataset_imputed = list(map(list, zip(*dataset_mv_imputed)))
            elif missing_imputation_method == 2:
                newAverages = []
                # message = " "
                dataset_imputed = perform_missing_value_imputation(new_data, newAverages, missing_imputation_method,
                                                                   user, jobid, pid)
            else:
                return [False, "Preprocessing raised the exception No imputation method provided. Please contact "
                               "info@insybio.com for more information", '']
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during imputation.".format(
                pid, jobid, user))
            return [False, "Preprocessing raised the exception during imputation. Please check if you selected the same"
                           " imputation method as the training step, or contact info@insybio.com for more information",
                    '']

        try:
            # Catch the cases where the user inputs only one element with a comma/newline at the end
            if '' in variables_for_normalization and not isinstance(variables_for_normalization, str) and \
                    not isinstance(variables_for_normalization, unicode):
                variables_for_normalization.pop()

            # Creating the list of indexes of the selected features
            variables_for_normalization_nums = list()
            if variables_for_normalization:
                if isinstance(variables_for_normalization, list):
                    for variable in variables_for_normalization:
                        try:
                            variable_index = new_features.index(variable)
                            variables_for_normalization_nums.append(variable_index)
                        except Exception:
                            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tThe biomarker(s) provided are not in the "
                                              "list of the biomarkers of the input file!".format(pid, jobid, user))
                            return [False, "The biomarker(s) provided gor normalization are not in the list of the "
                                           "biomarkers of the input file! Try again!", '']
                elif isinstance(variables_for_normalization, str):
                    try:
                        variable_index = new_features.index(variables_for_normalization)
                        variables_for_normalization_nums.append(variable_index)
                    except Exception:
                        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tThe biomarker(s) provided are not in the list of "
                                          "the biomarkers of the input file!".format(pid, jobid, user))
                        return [False, "The biomarker(s) provided gor normalization are not in the list of the "
                                       "biomarkers of the input file! Try again!", '']

            # print(variables_for_normalization_nums)
            # if the variables_for_normalization list isn't empty
            if variables_for_normalization and missing_imputation_method != 0:
                dataset_imputed = fix_negative_values(dataset_imputed, variables_for_normalization_nums)

                [dataset_imputed, message] = geometric_mean_normalization(
                    dataset_imputed, variables_for_normalization_nums, message, user, jobid, pid)

            # Perform normalization of dataset
            if normalization_method == 1:
                if maximums and minimums:
                    message += " "
                    # dataset_imputed = np.transpose(dataset_imputed)
                    normalized_mv_imputed_dataset = normalize_dataset(dataset_imputed, minimums, maximums,
                                                                      normalization_method, user, jobid, pid)
                    # normalized_mv_imputed_dataset = np.transp]]]]]]]
                    # 
                    # 
                    # ]]]]]]]   ]ose(normalized_mv_imputed_dataset)
                    dataset_imputed = normalized_mv_imputed_dataset
                else:
                    message += " No normalization took place."
                    inputs = dataset_imputed
            else:

                newMaximums = list()
                newMinimums = list()

                if has_negatives(dataset_imputed):
                    logging.error("PID:{}\tJOB:{}\tUSER:{}\tYour data contain negative values. Logarithmic "
                                  "normalization is not supported.!".format(pid, jobid, user))
                    return [False, "Your data contain negative values. Logarithmic normalization is not supported.! "
                                   "Try again!", '']

                # dataset_imputed = np.transpose(dataset_imputed)
                normalized_mv_imputed_dataset = normalize_dataset(dataset_imputed, newMinimums, newMaximums,
                                                                  normalization_method, user, jobid, pid)
                # normalized_mv_imputed_dataset = np.transpose(normalized_mv_imputed_dataset)
                dataset_imputed = normalized_mv_imputed_dataset

        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during "
                              "normalization.".format(pid, jobid, user))
            return [False, "Preprocessing raised an exception during normalization. Please contact info@insybio.com "
                           "for more information", '']

        # Print output message to info.txt file
        with open(output_folder_name + "info.txt", "w") as handle:
            handle.write(message)

        try:
            # write preprocessed data to file
            print_data(dataset_imputed, new_features, samples, output_folder_name, "preprocessed_dataset_{}.tsv".format(
                jobid))

        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during printing "
                              "data.".format(pid, jobid, user))
            return [False, "Preprocessing raised the exception during printing data. Please contact info@insybio.com "
                           "for more information", '']
        new_data = dataset_imputed
    return [True, new_data, new_features]


if __name__ == "__main__":
    input_dataset1 = sys.argv[1]
    variables_for_normalization_string1 = sys.argv[2]
    output_folder_name1 = sys.argv[3]
    output_dataset1 = sys.argv[4]
    missing_threshold1 = float(sys.argv[5])
    missing_imputation_method1 = int(sys.argv[6])
    normalization_method1 = int(sys.argv[7])
    filetype1 = int(sys.argv[8])
    has_features_header1 = int(sys.argv[9])
    has_samples_header1 = int(sys.argv[10])
    result = preprocess_data(input_dataset1, [], [], '', '', missing_imputation_method1, normalization_method1,
                             output_folder_name1, '1', variables_for_normalization_string1,
                             has_features_header1, has_samples_header1, 'test', 0, 0)
    print(result)
