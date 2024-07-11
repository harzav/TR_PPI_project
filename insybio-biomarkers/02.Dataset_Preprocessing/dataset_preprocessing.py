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
import configparser
import datetime
import logging
import math
import numpy as np
from math import sqrt
from sklearn import preprocessing


def initLogging():
    """
    Purpose: sets the logging configurations and initiates logging
    """
    config = configparser.ConfigParser()
    scriptPath = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
    scriptParentPath = os.path.abspath(os.path.join(scriptPath, os.pardir))
    configParentPath = os.path.abspath(os.path.join(scriptParentPath, os.pardir))
    config.read(configParentPath + '/insybio.ini')

    todaystr = datetime.date.today().strftime("%Y%m%d")
    logging.basicConfig(
        filename="{}biomarkers_reports_preprocessing_{}.log".format(config["logs"]["logpath"], todaystr),
        level=logging.DEBUG, format='%(asctime)s\t %(levelname)s\t%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')


def parse_data(data_filename, delimiter):
    """
    Parses data.

    Args:
        data_filename (string): dataset filename
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
                            data[num_of_lines - 1].append(float(line[j]))
                        else:
                            data[num_of_lines - 1].append('')
            num_of_lines += 1
    # logging.info('Data were successfully parsed!')
    return [proteins, data, samples]


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
                    if line[j] != '':
                        data[num_of_lines].append(float(line[j]))
                    else:
                        data[num_of_lines].append('')
            num_of_lines += 1
    # logging.info('Data were successfully parsed!')
    return [proteins, data]


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
                        data[num_of_lines - 1].append(float(line[j]))
                    except Exception:
                        data[num_of_lines - 1].append('')
            num_of_lines += 1
    # print('Data were successfully parsed!')
    return [data, samples]


def parse_only_dataset(dataset_filename, delimiter):
    """
    Parses a dataset which has no headers at all.

    Args:
        dataset_filename (string): the dataset filename
        delimiter: what delimiter has the input file (tab or comma)

    Returns:
        data (list): a list of lists with the data
        delimiter (string): the kind of delimiter with values "," or "\t"

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
    # logging.info('Data were successfully parsed!')
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
        head = next(handle)
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
            if dataset_initial[i][j] == '':
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


def perform_missing_values_imputation(dataset_initial, missing_imputation_method, output_folder_name, output_message,
                                      user='unknown', jobid=0, pid=0):
    """
    Perform missing values imputation.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        missing_imputation_method (integer): 1 for average imputation, 2 for KNN-impute
        output_folder_name: output folder name that will hold the averages for missing values imputation
        output_message: the output message that tells which imputation method was used
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: a list with the final dataset (list of lists) and the output message (string).
    """
    # missing values imputation
    averages = [0] * (len(dataset_initial))
    if missing_imputation_method == 1:
        # average imputation here
        num_of_non_missing_values = [0] * (len(dataset_initial))
        for j in range(len(dataset_initial)):
            for i in range((len(dataset_initial[0]))):
                if dataset_initial[j][i] != '':
                    # print(dataset_initial[j][i+1])
                    averages[j] += float(dataset_initial[j][i])
                    num_of_non_missing_values[j] += 1
            averages[j] = averages[j] / float(num_of_non_missing_values[j])

        write_one_dimensional_list_to_tab_delimited_file(
            averages, output_folder_name + 'averages_for_missing_values_imputation.txt')
        output_message += 'Average imputation method was used!\n'
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tAverage imputation method was used!".format(pid, jobid, user))
        for i in range(len(dataset_initial)):
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] == '':
                    dataset_initial[i][j] = averages[i]
        return [dataset_initial, output_message]
    else:
        # KNN-impute
        dataset_initial = list(map(list, zip(*dataset_initial)))
        for i in range(len(dataset_initial)):
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] == '':
                    dataset_initial[i][j] = np.NaN
        dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)), k=3)
        dataset = list(map(list, zip(*dataset)))
        output_message += 'KNN imputation method was used!\n'
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tKNN imputation method was used!".format(pid, jobid, user))
        return [dataset, output_message]


def normalize_dataset(dataset_initial, output_message, normalization_method, output_folder_name, user='unknown',
                      jobid=0, pid=0):
    """
    Normalize the dataset.

    Args:
        dataset_initial: the initial dataset (a list of lists)
        output_message: a string that holds the name of the normalization used
        normalization_method (integer): 1 for arithmetic sample-wise normalization, 2 for logarithmic normalization
        output_folder_name (string): the output folder name where the maximums and minimums will be stored
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns: if method 1 selected a list with the normalized dataset and the output message, else for method 2
            logged data are returned along with the output message.
    """
    if normalization_method == 1:
        # arithmetic sample-wise normalization
        maximums = [-1000.0] * (len(dataset_initial[0]))
        minimums = [1000.0] * (len(dataset_initial[0]))

        for i in range(len(dataset_initial)):
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] != '':
                    if float(dataset_initial[i][j]) > maximums[j]:
                        maximums[j] = float(dataset_initial[i][j])
                    if float(dataset_initial[i][j]) < minimums[j]:
                        minimums[j] = float(dataset_initial[i][j])
        write_one_dimensional_list_to_tab_delimited_file(maximums, output_folder_name + "maximums.txt")
        write_one_dimensional_list_to_tab_delimited_file(minimums, output_folder_name + "minimums.txt")
        # X = np.array(dataset_initial)
        # maximums = np.array(maximums)
        # minimums = np.array(minimums)
        # X_min_max = (X - minimums)/(maximums - minimums)
        # dataset_initial = X_min_max
        for i in range(len(dataset_initial)):
            for j in range((len(dataset_initial[0]))):
                if dataset_initial[i][j] != '':
                    if maximums[j] - minimums[j] == 0:
                        dataset_initial[i][j] = dataset_initial[i][j]
                    else:
                        dataset_initial[i][j] = 0 + (float(dataset_initial[i][j]) - minimums[j]) / float(
                            maximums[j] - minimums[j])
        output_message += 'Arithmetic normalization was used!\n'
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tArithmetic normalization was used!".format(pid, jobid, user))
        return [dataset_initial, output_message]
    else:
        logged_data = list()
        for i in range(len(dataset_initial)):
            # print('i='+str(i))
            logged_data.append([])
            for j in range(len(dataset_initial[0])):
                # print('j='+str(j))
                if dataset_initial[i][j] == '':
                    logged_data[i].append('')
                else:
                    if dataset_initial[i][j] == 0:
                        logged_data[i].append(0)
                    else:
                        logged_data[i].append(math.log2(dataset_initial[i][j]))
        output_message += 'Logarithmic normalization was used!\n'
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tLogarithmic normalization was used!".format(pid, jobid, user))
        return [logged_data, output_message]


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
                    raise ValueError(
                        "Error in geometric normalization. Gene {} contains zero values. Choose different gene.".format(
                            j))
                else:
                    geo_mean *= float(dataset_imputed[i][j])
        geo_mean = geo_mean ** (1. / len(selected_features))
        geometric_means.append(geo_mean)  # different geo_mean per sample
        geo_mean = 1
    print(geometric_means)
    for i in range(len(dataset_imputed)):
        for j in range(len(dataset_imputed[0])):
            if dataset_imputed[i][j] != '':
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
                if dataset_initial[i][j] != '':
                    dataset[markers[i]].append(float(dataset_initial[i][j]))
                else:
                    dataset[markers[i]].append(0)
        else:
            dict_of_occurences[markers[i]] += 1
            for j in range(len(dataset_initial[0])):
                if dataset_initial[i][j] != '':
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


def preprocess_data(input_dataset, variables_for_normalization_string, output_folder_name, output_dataset,
                    missing_threshold, missing_imputation_method, normalization_method, filetype, has_features_header,
                    has_samples_header, user='unknown', jobid=0, pid=0):
    """
    A script that preprocesses the data.

    Args:
        input_dataset: the initial dataset to be preprocessed
        variables_for_normalization_string: the string with the names of the selected genes that with which the
        geometric mean will be calculated, separated with commas or "\n"
        output_folder_name: the output folder name
        output_dataset: the preprocessed dataset
        missing_threshold: missing threshold
        missing_imputation_method (integer): 0 for no imputation, 1 for average imputation, 2 for KNN-impute
        normalization_method (integer): 0 for no normalization, 1 for arithmetic sample-wise normalization, 2 for
        logarithmic normalization
        filetype: if it is 7 then it doesn't have a samples header
        has_features_header (integer): 1 if it has, 0 if it doesn't have
        has_samples_header (integer): 1 if it has, 0 if it doesn't have
        user (string): THis job's username
        jobid (integer): this job's id in biomarkers_jobs table
        pid (integer): this job's PID

    Returns:
        output_dataset: the preprocessed dataset
    """
    # initLogging()
    try:
        # Parsing
        if not os.path.exists(output_folder_name):
            os.makedirs(output_folder_name)

        # Find delimiter
        delim = find_delimiter(input_dataset)

        if filetype != 7:
            if has_features_header and has_samples_header:
                markers, data, samples = parse_data(input_dataset, delim)

            elif has_features_header and not has_samples_header:
                markers, data = new_parse_data(input_dataset, delim)
                samples = create_samples_list(data)

            elif not has_features_header and has_samples_header:
                data, samples = parse_data_with_only_samples(input_dataset, delim)
                markers = create_feature_list(data)

            else:  # has nothing
                data = parse_only_dataset(input_dataset, delim)
                markers = create_feature_list(data)
                samples = create_samples_list(data)
        else:
            if has_features_header and has_samples_header:
                markers, data, samples = parse_data(input_dataset, delim)

            elif has_features_header and not has_samples_header:
                markers, data = new_parse_data(input_dataset, delim)
                samples = create_samples_list(data)

            elif not has_features_header and has_samples_header:
                data, samples = parse_data_with_only_samples(input_dataset, delim)
                markers = create_feature_list(data)

            else:  # has nothing
                data = parse_only_dataset(input_dataset, delim)
                markers = create_feature_list(data)
                samples = create_samples_list(data)

        variables_for_normalization = parse_selected_features_string(variables_for_normalization_string)

    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during parsing.".format(
            pid, jobid, user))
        return [0, "Preprocessing raised an exception during parsing the Dataset. Please contact info@insybio.com for"
                   " more information"]

    # Saving the length of features
    with open(output_folder_name + "length_of_features.txt", "w") as handle:
        handle.write(str(len(data)))

    try:
        output_message = ""
        # Filter proteomics dataset
        [dataset_filtered, markers, output_message] = filter_proteomics_dataset(data, markers, missing_threshold,
                                                                                output_message, output_folder_name,
                                                                                user, jobid, pid)
    # print_data(dataset_filtered, markers, samples, output_folder_name, "data_mv_filtered.txt")
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during filtering.".format(
            pid, jobid, user))
        return [0, "Preprocessing raised an exception during Dataset filtering. Please contact info@insybio.com for"
                   " more information"]

    try:
        # Perform missing values imputation
        if missing_imputation_method != 0:
            [dataset_imputed, output_message] = perform_missing_values_imputation(dataset_filtered,
                                                                                  missing_imputation_method,
                                                                                  output_folder_name, output_message,
                                                                                  user, jobid, pid)
        # print_data(dataset_imputed, markers, samples, output_folder_name, "data_mv_imputed.txt")
        else:
            dataset_imputed = dataset_filtered
            output_message += "No imputation method was used!\n"
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during imputation.".format(
            pid, jobid, user))
        return [0, "Preprocessing raised the exception during imputation. Please contact info@insybio.com for more "
                   "information"]

    try:
        # Catch the cases where the user inputs only one element with a comma/newline at the end
        if '' in variables_for_normalization and not isinstance(variables_for_normalization, str) and not isinstance(
                variables_for_normalization, unicode):
            variables_for_normalization.pop()

        # Creating the list of indexes of the selected features
        variables_for_normalization_nums = list()
        if variables_for_normalization:
            if isinstance(variables_for_normalization, list):
                for variable in variables_for_normalization:
                    try:
                        variable_index = markers.index(variable)
                        variables_for_normalization_nums.append(variable_index)
                    except Exception:
                        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tThe biomarker(s) provided are not in the list of "
                                          "the biomarkers of the input file!".format(pid, jobid, user))
                        return [0, "The biomarker(s) provided gor normalization are not in the list of the biomarkers "
                                   "of the input file! Try again!"]
            elif isinstance(variables_for_normalization, str):
                try:
                    variable_index = markers.index(variables_for_normalization)
                    variables_for_normalization_nums.append(variable_index)
                except Exception:
                    logging.exception("PID:{}\tJOB:{}\tUSER:{}\tThe biomarker(s) provided are not in the list of "
                                      "the biomarkers of the input file!".format(pid, jobid, user))
                    return [0, "The biomarker(s) provided gor normalization are not in the list of the biomarkers "
                               "of the input file! Try again!"]

        # print(variables_for_normalization_nums)
        # if the variables_for_normalization list isn't empty
        if variables_for_normalization and missing_imputation_method != 0:
            dataset_imputed = fix_negative_values(dataset_imputed, variables_for_normalization_nums)

            [dataset_imputed, output_message] = geometric_mean_normalization(
                dataset_imputed, variables_for_normalization_nums, output_message)

        try:
            # Average duplicate measurements & outlier detection
            if missing_imputation_method != 0:
                [dataset_imputed, markers] = average_duplicate_measurements(dataset_imputed, markers, user, jobid, pid)
                output_message += "Duplicate measurements have been averaged successfully!\n"
                # print_data(dataset_imputed, markers, samples, output_folder_name,
                # "dataset_averaged_duplicate_measurements.csv")
                pca_data = outlier_detection(dataset_imputed, output_folder_name)
            # data_transp=list(map(list, zip(*pca_data)))
            else:
                output_message += "Duplicate measurements have not been averaged!\n"
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during averaging.".format(
                pid, jobid, user))
            return [0, "Preprocessing raised the exception during averaging. Please contact info@insybio.com for more "
                       "information"]

        # Perform normalization of dataset
        if normalization_method != 0:
            if normalization_method == 2:
                if has_negatives(dataset_imputed):
                    logging.exception("PID:{}\tJOB:{}\tUSER:{}\tYour data contains negative values. Logarithmic "
                                      "normalization is not supported.!".format(pid, jobid, user))
                    return [0, "Your data contains negative values. Logarithmic normalization is not supported! "
                               "Fix the values and try again!"]
            dataset_imputed = np.transpose(dataset_imputed)
            [dataset_imputed, output_message] = normalize_dataset(dataset_imputed, output_message, normalization_method,
                                                                  output_folder_name, user, jobid, pid)
            dataset_imputed = np.transpose(dataset_imputed)
        # print_data(dataset_imputed,markers,samples,output_folder_name, "dataset_imputed_normalized_only2.csv")
        else:
            dataset_imputed = dataset_imputed
            output_message += "No normalization method was used!\n"
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during normalization.".format(
            pid, jobid, user))
        return [0, "Preprocessing raised an exception during normalization. Please contact info@insybio.com for more "
                   "information"]

    # Print output message to info.txt file
    with open(output_folder_name + "info.txt", "w") as handle:
        handle.write(output_message)

    try:
        # write preprocessed data to file
        print_data(dataset_imputed, markers, samples, output_folder_name, output_dataset)

    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tPreprocessing raised the exception during printing data.".format(
            pid, jobid, user))
        return [0, "Preprocessing raised the exception during printing data. Please contact info@insybio.com for more "
                   "information"]
    return [1, "Job completed successfully."]


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
    result = preprocess_data(input_dataset1, variables_for_normalization_string1, output_folder_name1, output_dataset1,
                             missing_threshold1, missing_imputation_method1, normalization_method1, filetype1,
                             has_features_header1, has_samples_header1)
    print(result)
