"""
Script that chooses which subscript to run according to the selection_flag given.

Example run for multi class script:
python3 biomarker_discovery_script_selection_new_backend.py classification_multiple_labels/Input/peptides_dataset.txt
  classification_multiple_labels/Input/peptides_labels.txt "1,100,1" "" 0 0 0.3 50 100 0.01 0 0.9 5 8 0 1 Output_Folder/
python3 biomarker_discovery_script_selection_new_backend.py
classification_multiple_labels/Input/output_dataset_strings_multi.txt
classification_multiple_labels/Input/example_labels.txt "1,100,1" "" 0 0 0.3 50 100 0.01 0 0.9 5 8 0 0 Output_Folder/

Example of run for regression script:
python3 biomarker_discovery_script_selection_new_backend.py regression/Input/total_drugs_dataset.txt
regression/Input/total_severity_regression_labels.txt "1,10,1" "" 1 1 0.3 50 100 0.01 0 0.9 5 8 0 0 Output_Folder/

Example run for two class script:
python3 biomarker_discovery_script_selection_new_backend.py
classification_two_classes/Input/combined_stroke_inputs_nolabels.txt
classification_two_classes/Input/combined_stroke_labels.txt "1,10,10,1" "" 2 1 0.3 50 100 0.01 0 0.9 5 8 0 0
Output_Folder/
python3 biomarker_discovery_script_selection_new_backend.py
classification_two_classes/Input/output_dataset_single_with_geoFinal_0.1_1_2.txt
classification_two_classes/Input/example_labels.txt "1,10,10,1" "ACTB,VIME,APOE,TLN1,CO6A3" 2 1 0.3 50 100 0.01 0 0.9 5
8 1 1 Output_Folder/
"""

import sys
import time
import os
import random
import numpy as np
import csv
import configparser
import datetime
import logging
import itertools
import json
from shutil import copyfile

import ensemble_CVD_regression.ensemble_CVD_regressor as ensemble_regression
import ensemble_CVD_twoclass.ensemble_CVD_binary_classifierADABOOST as ensemble_twoclass
import ensemble_CVD_multiclass.ensemble_CVD_multiclass_classifierAdaboost as ensemble_multiclass
import ensemble_CVD_multiclass.preprocessing_data_multiclass as preprocessing_multi
import ensemble_CVD_regression.preprocessing_regression as preprocessing_regression
import ensemble_CVD_twoclass.preprocessing_data_two_class as preprocessing_twoclass
from stripper import parse_data, print_data
from plotting import draw_plots

from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor


def initLogging(selection_flag):
    """
    Purpose: sets the logging configurations and initiates logging used only debugging, if the script is called as main
    """
    config = configparser.ConfigParser()
    scriptPath = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
    scriptParentPath = os.path.abspath(os.path.join(scriptPath, os.pardir))
    configParentPath = os.path.abspath(os.path.join(scriptParentPath, os.pardir))
    config.read(configParentPath + '/insybio.ini')

    todaystr = datetime.date.today().strftime("%Y%m%d") + "_" + str(time.time())
    if selection_flag == 0:
        mystring = "multi"
    elif selection_flag == 1:
        mystring = "regr"
    else:
        mystring = "twoclass"
    logging.basicConfig(
        filename="{}biomarkers_reports_modeller_{}_{}.log".format(config["logs"]["logpath"], mystring, todaystr),
        level=logging.INFO, format='%(asctime)s\t %(levelname)s\t%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')


def parsing_oneline(oneline_filename):
    """
    Parses a file with one line, with values separated with tabs. Also works with data separated with newlines
    :param oneline_filename: a tab separated file
    :return: a list
    """
    delim = find_delimiter_labels(oneline_filename)
    line2list = list()
    with open(oneline_filename, 'r') as linefile:
        for line in linefile:
            line2list.extend(line.strip().split(delim))
    return line2list


def parsing_features(feature_file):
    """
       Parses a feature file, with feature names as columns and each line has value 1 if the feature is selected,
        else has 0
       :param feature_file: a tab separated file
       :return: a list
       """
    line2list = list()
    with open(feature_file, 'r') as maximums_file:
        for i, line in enumerate(maximums_file):
            if i == 0:
                continue
            words = line.strip().split(",")
            for j, word in enumerate(words):
                if word == '1':
                    line2list.append(j)
    return list(set(line2list))


def parse_selected_features(features_filename, user='unknown', jobid=0, pid=0):
    """
    Parses the selected features filename.
    :param features_filename: the selected features filename, one line tab separated values
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: features (list): the list of the selected features
    """
    try:
        features = list()
        with open(features_filename, "r") as features_fid:
            for line1 in features_fid:
                words = line1.split("\t")
                for i in range(len(words)):
                    labels.append(words[i].strip())
            # print("Selected features file was successfully parsed!")
            # logging.info("Selected features file was successfully parsed!")
            return features
    except Exception:
        # print(e)
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during parsing the selected"
                          " features.".format(pid, jobid, user))
        return None


def print_data_all(data, markers, labels, folder_name, filename, user='unknown', jobid=0, pid=0):
    """
    Writes data and labels to a file.
    :param data: input data (list of lists)
    :param markers: input biomarkers (list)
    :param labels: input labels (list)
    :param folder_name: output folder
    :param filename: output filename
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """
    try:
        with open(folder_name + filename, 'w') as file:
            message = ''
            for i in range(len(data[0])):
                message = message + '\t' + labels[i]
            message += '\n'
            for i in range(len(data)):
                message += markers[i] + '\t'
                message += '\t'.join(map(str, data[i]))
                # for j in range(len(data[0])):
                #     message += '\t' + str(data[i][j])
                message += '\n'
            file.write(message)
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tData have been written to file successfully.".format(pid, jobid,
                                                                                                        user))
            return True
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during parsing the selected"
                          " features.".format(pid, jobid, user))
        return False


def find(name, path, user='unknown', jobid=0, pid=0):
    """
    Locates a file in a given path and returns the path. If not found returns None
    :param name: name of file
    :param path: path where to look for the file
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: if found filepath, else None
    """
    try:
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during locating file"
                          " path.".format(pid, jobid, user))
        return False


def parse_goal_significances(goal_significances_filename, user='unknown', jobid=0, pid=0):
    """
    Parses the goal significances filename.
    :param goal_significances_filename: (string) the goal significances filename
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: goal_significances: (list of floats) Includes the significances (values from 0 to 1) of the individual goals
    """
    try:
        with open(goal_significances_filename, "r") as goal_significances_fid:

            number_of_lines = 0
            for line in goal_significances_fid:
                words = line.split("\t")
                for word in words:
                    goal_significances.append(float(word.strip()))
                number_of_lines += 1
        logging.info("Goal significances have been parsed successfully.")
        return goal_significances
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during locating file"
                          " path.".format(pid, jobid, user))
        return False


def get_min_max(dataset_filename, min_values, max_values, user='unknown', jobid=0, pid=0):
    """
    Creates the min_values and max_values lists.
    :param dataset_filename: (string) the dataset filename
    :param min_values: (list) the initial min_values list
    :param max_values: (list) the initial max_values list
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: min_values: the final min_values list, max_values: the final max_values list
    """
    try:
        with open(dataset_filename, "r") as dataset_fid:
            number_of_lines = 0
            for _ in dataset_fid:
                number_of_lines += 1
                min_values.append(0.0)
                max_values.append(1.0)
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tMin, max lists have been created successfully.".format(pid, jobid,
                                                                                                          user))
            return min_values, max_values
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during min max creation.".format(
            pid, jobid, user))
        return False, False


def write_one_line_tab_delimited_file(proteins, filename, user='unknown', jobid=0, pid=0):
    """
    Writes an one dimensional list to an one line tab delimited file.
    :param proteins: (list): the one dimensional list
    :param filename: (string): the filename where the data are being written
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """
    try:
        with open(filename, 'w') as handle:
            handle.write("\t".join(proteins))
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tAn exception has been raised during I/O in a file.".format(
            pid, jobid, user))


def strip_dataset(dataset_filename, output_folder, delimiter, user='unknown', jobid=0, pid=0):
    """
    Stripping a dataset from its features and sample headers.
    :param dataset_filename: (string): the name of the dataset file
    :param output_folder: (string): the name of the output folder
    :param delimiter: (string): the kind of delimiter with values "," or "\t"
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: final_path: the final path where the file is stored
    """

    features, data, samples = parse_data_generic(dataset_filename, delimiter, user, jobid, pid)
    dataset_filename = "input_dataset_stripped.txt"
    print_data(data, output_folder, dataset_filename)
    logging.info("PID:{}\tJOB:{}\tUSER:{}\tData have been stripped from markers and samples successfully.".format(
        pid, jobid, user))
    final_path = output_folder + dataset_filename
    return final_path, features


def parse_selected_features_string(astring):
    """
    Parses a string and strips it from commas or newline characters.
    :param astring: the input string with comma separated or newline separated values
    :return: A list with the substrings of the original string.
    """

    # print(astring)
    if "," in astring:
        return astring.split(",")
    elif "\\n" in astring:
        return astring.split("\\n")
    else:
        # raise ValueError("The string doesn't contain comma separated values or newline separated values !")
        return astring


def parse_data_generic(data_filename, delimiter, user='unknown', jobid=0, pid=0):
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
                        data[num_of_lines].append('')
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
                        data[num_of_lines - 1].append('')
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
    logging.info('PID:{}\tJOB:{}\tUSER:{}\tData were successfully parsed!'.format(pid, jobid, user))
    return data


def create_feature_list(dataset):
    """
    Creates a feature list with dummy names for a given dataset.
    :param dataset: (list): a list of lists with the data values
    :return: a one dimensional list of length equal to a column's length of the dataset with strings "Feature_0",
     "Feature_1", etc.
    """

    n = len(dataset)
    return ["Feature_" + str(i) for i in range(n)]


def create_samples_list(dataset):
    """
    Creates a samples list with dummy names for a given dataset.
    :param dataset: (list): a list of lists with the data values
    :return: a one dimensional list of length equal to a line's length of the dataset with strings "Sample_0",
     "Sample_1", etc.
    """
    n = len(dataset[0])
    return ["Sample_" + str(i) for i in range(n)]


def write_oneline_to_newline_delimited_file(proteins, filename):
    """
    Writes the data of an one dimensional list to a file with newline delimited data.

    Args:
        proteins (list): the one dimensional list
        filename (string): the output file
    """
    with open(filename, "w") as handle:
        for i in range(len(proteins)):
            handle.write(proteins[i])
            if i < len(proteins) - 1:
                handle.write("\n")


def new_print_data(data, markers, folder_name, filename, user='unknown', jobid=0, pid=0):
    """
    Writes data and labels to a file.
        :param data: input data (list of lists)
        :param markers: input biomarkers (list)
        :param folder_name: output folder
        :param filename: output filename
        :param user: this job's user
        :param jobid: this job's id
        :param pid: this job's pid
    :return:  True if executed successfully, else False
    """

    try:
        with open(folder_name + filename, 'w') as file:
            message = ''
            for i, line in enumerate(data):
                message += markers[i] + '\t'
                message += '\t'.join([str(x) for x in line])
                message += '\n'
            file.write(message)
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tError when writing data and labels to file.".format(
            pid, jobid, user))
        return False


def find_delimiter(dataset_filename):
    """
    Figures out which delimiter is being used in given dataset.
    :param dataset_filename: (string): the dataset filename
    :return: "," if CSV content, "\t" if TSV content.
    """

    with open(dataset_filename, 'r') as handle:
        line = next(handle)
        #line = next(handle)
    if "\t" in line:
        return "\t"
    elif "," in line:
        return ","
    elif "," and "\t" in line:  # The case where the comma is the decimal separator (greek system)
        return "\t"


def reconstruct_testset_with_markers(stripped_testset_filename, markers, folder_name, output_filename,
                                     user='unknown', jobid=0, pid=0):
    """
    Reconstructs the testset filename with the markers. Samples are not needed.
    :param stripped_testset_filename: (string): the stripped testset filename
    :param markers: the list with the markers
    :param folder_name: the name of the output folder
    :param output_filename: the name of the output filename
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """

    data = parse_only_dataset(stripped_testset_filename, "\t", user, jobid, pid)
    new_print_data(data, markers, folder_name, output_filename, user, jobid, pid)


def create_MQ_files(dataset, markers, labels, output_filename, output_folder, user='unknown', jobid=0, pid=0):
    """
    Creates the MQ files of given dataset, samples, markers and labels.
    :param dataset: (list): list of lists of input data, with markers as rows and samples as columns
    :param markers: list): list of marker names
    :param labels: (list): list of input labels
    :param output_filename: (string): the prefix of the name of the output file
    :param output_folder: (string): the name of the output folder
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return: nothing: prints data to file
    """

    data_transp = np.transpose(dataset)
    unique_labels = sorted(list(set(labels)))
    for tag in unique_labels:
        new_dataset = list()
        # new_samples = list()
        for i, label in enumerate(labels):
            if tag == label:
                new_dataset.append(data_transp[i])
        # new_samples.append(samples[i])
        new_dataset = np.transpose(new_dataset)
        new_print_data(new_dataset, markers, output_folder, output_filename + tag + ".tsv", user, jobid, pid)


def create_selected_biomarkers_file(data1, data2, gene_names, logged_flag, output_folder, filename_result,
                                    user='unknown', jobid=0, pid=0):
    """
    Creates the selected biomarkers file by reading data from two MQ files
    :param data1: (list): the first input MQ dataset
    :param data2: (list): the second input MQ dataset
    :param gene_names: (list): the gene names
    :param logged_flag: (int): 0 for not logarithmic, 1 for logarithmic
    :param output_folder: (string): the output folder
    :param filename_result: (string): the name of the output file
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:  nothing, prints data to file named filename_result.
    """

    avgs1 = list()
    avgs2 = list()
    fold_changes = list()
    for i in range(0, len(data1)):
        avgs1.append(float(sum(data1[i])) / max(len(data1[i]), 1))
        avgs2.append(float(sum(data2[i])) / max(len(data2[i]), 1))
        if logged_flag == 0:
            if avgs1[i] == 0:
                fold_changes.append('')
            else:
                fold_changes.append(avgs2[i] / avgs1[i])
        else:
            fold_changes.append(avgs2[i] - avgs1[i])

    with open(output_folder + filename_result, "w") as output_file:
        for i in range(len(avgs1)):
            output_file.write(gene_names[i] + "\t" + str(avgs1[i]) + "\t" + str(avgs2[i]) + "\t"
                              + str(fold_changes[i]) + "\n")


def get_pairs_from_list(alist):
    """
    Gets all pairs from a list and returns them in a list of unique pairs.
    :param alist: (list): the input list
    :return: all_pairs(list): the list of unique pairs
    """

    all_pairs = list()
    for pair in itertools.combinations(alist, 2):
        all_pairs.append(pair)
    return all_pairs


def select_modeller(dataset, labels, features, goal_significances, output_folder, selection_flag, population,
                    generations, mutation_probability, arithmetic_crossover_probability,
                    two_points_crossover_probability, num_of_folds, user='unknown', jobid=0, pid=0, thread_num=1):
    """
    Selects the modeller we want to run.
    :param dataset: (list) 2d list with all per sample feature values
    :param labels: (list) list of per sample labels
    :param features: (list) list of all deataset features names
    :param goal_significances: (list of floats) Includes the significances (values positive integers) of the
    individual goals
    :param output_folder: (string) output folder
    :param selection_flag: (integer) 0 for multiclass, 1 for regression and 2 for two classes
    :param population:(integer) (default: 50) it is the number of individual solutions which are evolved on parallel
    :param generations: (integer) (default: 100) it is the maximum number of generations which we allow for the
    population to get evolved
    :param mutation_probability: (float) (default: 0.01) it is the probability for applying gaussian mutation operator
    :param arithmetic_crossover_probability: (float) (default: 0) it is the probability (multiplied with 100) for
    using  arithmetic crossover operator
    :param two_points_crossover_probability: (float) (default: 0.9) it is the probability (multiplied with 100) for
    using two points crossover operator
    :param num_of_folds: (integer) (default: 5) the number of k folds in k-fold cross validation
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :param thread_num: number of available threads used for parallelism
    :return: True if finished successfully, else False
    """
    result = [True, '']
    if selection_flag == 0:
        result = ensemble_multiclass.main_only_training(
            output_folder, dataset, labels, features, population, generations, num_of_folds, mutation_probability,
            arithmetic_crossover_probability, two_points_crossover_probability, thread_num, goal_significances,
            user, jobid, pid)
    elif selection_flag == 1:

        result = ensemble_regression.main_only_training(
            output_folder, dataset, labels, features, population, generations, num_of_folds, mutation_probability,
            arithmetic_crossover_probability, two_points_crossover_probability, thread_num, goal_significances, user,
            jobid, pid)
    elif selection_flag == 2:

        result = ensemble_twoclass.main_only_training(
            output_folder, dataset, labels, features, population, generations, num_of_folds, mutation_probability,
            arithmetic_crossover_probability, two_points_crossover_probability, thread_num, goal_significances,
            user, jobid, pid)
    if result[0]:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tModeller has been selected successfully.".format(pid, jobid, user))

    else:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tModeller has been selected unsuccessfully. {}".format(pid, jobid, user,
                                                                                                      result[1]))
        raise Exception(result[1])


def select_splitter(dataset, labels, labelnames, features, samples, filtering_percentage, selection_flag, output_folder,
                    user='unknown', jobid=0, pid=0):
    """
    Selects the splitter that will split the data.
    :param dataset: dataset with feature values for each sample
    :param labels: labels for each sample
    :param labelnames: list of label names if alphanumeric
    :param features: list of dataset features
    :param samples: samples names
    :param filtering_percentage: the split percentage
    :param selection_flag: 0 for multi, 1 for regression, 2 for two class
    :param output_folder: the output folder where the training_set, test_set, training_set_labels and test_set_labels
     will be stored
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :return:
    """
    res = [True, ""]

    if selection_flag == 0:
        res = preprocessing_multi.splitter_multiclass(dataset, labels, labelnames, features, samples,
                                                      filtering_percentage, output_folder, user, jobid, pid)
    elif selection_flag == 1:
        res = preprocessing_regression.splitter_regression(dataset, labels, features, samples, filtering_percentage,
                                                           output_folder, user, jobid, pid)
    elif selection_flag == 2:
        res = preprocessing_twoclass.splitter_twoclass(dataset, labels, labelnames, features, samples,
                                                       filtering_percentage, output_folder, user, jobid, pid)

    if res[0]:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tSplitter has been selected successfully.".format(pid, jobid, user))
        return [True, res[1], res[2]]
    else:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tException during run of splitter:{}".format(pid, jobid, user, res[1]))
        return [False, "Exception during dataset splitting into Training and Test sets", '']


def run_model_and_splitter_selectors(dataset_filename, labels_filename, filtering_percentage, selection_flag,
                                     split_dataset_flag, goal_significances_string, selected_comorbidities_string,
                                     filetype, has_feature_headers, has_sample_headers, output_folder, logged_flag=0,
                                     population=50, generations=100, mutation_probability=0.01,
                                     arithmetic_crossover_probability=0.0, two_points_crossover_probability=0.9,
                                     num_of_folds=5, user='unknown', jobid=0, pid=0, thread_num=1):
    """
    Runs the main functions, select_splitter and select_modeller. Also parses goal significances, creates output folder
     and deletes the selected columns (commorbidities) from the input dataset, if commorbidities are being provided.
     If not, modellers are being run on the full dataset, unless if the user chooses to split it by some percentage.
     Also, this function strips the input dataset from the markers and the samples (if given) before feeding the
     dataset into the modeller.
    :param dataset_filename: (string): the filename of the dataset txt file (TSV or CSV). The rows are the features and
     the columns are the samples.
    :param labels_filename: the filename of the labels TSV or CSV file
    :param filtering_percentage: the percentage of the total dataset that will be contained in the test set
    :param selection_flag: 0 for multiclass classification, 1 for regression, 2 for two class classification
    :param split_dataset_flag: 1 if the initial dataset is going to be splitted to test set and training set, 0 if not
    :param goal_significances_string: the string with the goal significances
    :param selected_comorbidities_string: the selected comorbidities names, separated by commas or newline characters;
     they won't be considered as inputs
    :param filetype: input dataset's filetype,
    :param has_feature_headers: 1 if it has feature headers, 0 if it doesn't have
    :param has_sample_headers: 1 if it has sample headers, 0 if it doesn't have
    :param output_folder: the output folder
    :param logged_flag: 0 if data not logged, 1 if logged
    :param population: (default: 50): it is the number of individual solutions which are evolved on parallel
    :param generations: (default: 100): it is the maximum number of generations which we allow for the population to
    get evolved
    :param mutation_probability: (default: 0.01): it is the probability for applying gaussian mutation operator
    :param arithmetic_crossover_probability: (default: 0): it is the probability (multiplied with 100) for using
    arithmetic crossover operator
    :param two_points_crossover_probability: (default: 0.9): it is the probability (multiplied with 100) for using
    two points crossover operator
    :param num_of_folds: (default: 5): the number of k folds in k-fold cross validation
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's pid
    :param thread_num: number of available threads, used for parallelism script in training
    :return: When run this function outputs the following files are produced as a result:

        metrics.txt: the accuracy metrics (accuracy, sensitivity, specificity for classification and mse accuracy for
        regression)
        features_list.txt: the selected features, by their indexes (starting from 0)
        best_solutions.txt:
        best_solution.txt:
        best_performance.txt: the best performance per generation
        average_performance.txt: the average performance per generation
        final_solutions.txt: the final solutions
        selected_normalized.txt (deprecated, to be deleted)
        selected_normalized_missing_values_completed.txt (deprecated, to be deleted)
        number_of_svs.txt: the average normalized number of support vectors (model complexity)
        average_performance_graph.png: the graph of the average performance per generation
        best_performance_graph.png: the graph of the best performance per generation
    """

    # Initialization

    random.seed()
    """min_values = [0.0, 0.001, 0.001, 0.001]
    max_values = [2.0, 1000.0, 1000.0, 0.8]"""

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Converting the string of a list to a list.
    goal_significances = parse_selected_features_string(goal_significances_string)
    goal_significances += [0]*(9-len(goal_significances))

    # Find delimiter for input dataset
    delim = find_delimiter(dataset_filename)

    # Parse dataset and if there exists a commorbidities file, then omit those columns from the dataset
    # If a non empty selected_features.txt files exists then a commorbidities file exists.

    try:
        if has_feature_headers and has_sample_headers:
            proteins, data, samples = parse_data_generic(dataset_filename, delim, user, jobid, pid)

        elif has_feature_headers and not has_sample_headers:
            proteins, data = new_parse_data(dataset_filename, delim, user, jobid, pid)
            samples = create_samples_list(data)

        elif not has_feature_headers and has_sample_headers:
            data, samples = parse_data_with_only_samples(dataset_filename, delim, user, jobid, pid)
            proteins = create_feature_list(data)

        else:  # has nothing
            data = parse_only_dataset(dataset_filename, delim, user, jobid, pid)
            proteins = create_feature_list(data)
            samples = create_samples_list(data)

        write_one_line_tab_delimited_file(proteins, output_folder + "markers.txt")
        markers_filename = output_folder + "markers.txt"

        with open(output_folder + "features_list.txt", 'w') as handle:
            handle.write(",".join(proteins))

        # Saving the length of the features
        with open(output_folder + "length_of_features_from_training.txt", "w") as handle:
            handle.write(str(len(data)))

    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during parsing of dataset.".format(pid, jobid, user))
        return [0, "Exception during parsing of dataset: {}".format(e)]

    has_outliers = checkdata(data, proteins, user, jobid, pid)
    if has_outliers[0]:
        return [0, has_outliers[1]]
    
    labels = parse_labels(labels_filename, len(data[0]), selection_flag, user, jobid, pid)
    labels_alphanumeric = parsing_oneline(labels_filename)
    if not labels[0]:
        return [0, labels[1]]

    if selected_comorbidities_string:
        try:

            selected_comorbidities = parse_selected_features_string(selected_comorbidities_string)

            data = np.transpose(data)
            new_data = [[] for _ in range(len(data))]
            for i in range(len(data)):
                for j in range(len(data[0])):
                    if not proteins[j] in selected_comorbidities:
                        new_data[i].append(data[i][j])
            new_data = np.transpose(new_data)
            data = new_data
            # Create the new proteins excluding those that are selected
            # Assumes that the selected_comorbidities contains the names of the selected features, not indexes !
            new_proteins = [x for x in proteins if x not in selected_comorbidities]
            write_one_line_tab_delimited_file(new_proteins, output_folder + "new_markers.txt")
            markers_filename = output_folder + "new_markers.txt"

            # Print new data without the selected features to a file
            print_data_all(new_data, new_proteins, samples, output_folder, "dataset_wout_selected.txt", user, jobid,
                           pid)
            dataset_filename = output_folder + "dataset_wout_selected.txt"
        except Exception as e:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during parsing selected features string.".format(
                pid, jobid, user))
            return [0, "Exception during parsing selected features string: {}".format(e)]

    # Strip dataset from marker and sample names
    else:  # In case commorbidities are not provided
        if not has_sample_headers and has_feature_headers:
            print_data_all(data, proteins, samples, output_folder, "dataset_with_samples.txt", user, jobid, pid)
            dataset_filename = output_folder + "dataset_with_samples.txt"

        if not has_feature_headers and has_sample_headers:
            print_data_all(data, proteins, samples, output_folder, "dataset_with_features.txt", user, jobid, pid)
            dataset_filename = output_folder + "dataset_with_features.txt"

        if not has_feature_headers and not has_sample_headers:
            print_data_all(data, proteins, samples, output_folder, "dataset_with_features_and_samples.txt", user, jobid,
                           pid)
            dataset_filename = output_folder + "dataset_with_features_and_samples.txt"
    try:
        delim = find_delimiter(dataset_filename)
        dataset_filename, proteins = strip_dataset(dataset_filename, output_folder, delim, user, jobid, pid)
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during stripping of dataset.".format(pid, jobid, user))
        return [0, "Exception during stripping of dataset: {}".format(e)]

    """min_values, max_values = get_min_max(dataset_filename, min_values, max_values, user, jobid, pid)
    if not min_values:
        return [0, "Exception during creation of min, max lists"]"""

    # Split the dataset to training and test set if the user wants
    if split_dataset_flag == 1:

        result = select_splitter(data, labels[0], labels_alphanumeric, proteins, samples, filtering_percentage,
                                 selection_flag, output_folder, user, jobid, pid)
        if result[0]:
            data = result[1]
            labels = [result[2]]
        else:
            return [0, result[1]]

        # Check if testset and test labels are empty files due to small dataset and small filtering percentage
        if os.stat(output_folder + "test_dataset.txt").st_size == 0:

            logging.error("PID:{}\tJOB:{}\tUSER:{}\tPlease increase the filtering percentage! Empty testset "
                          "was produced!".format(pid, jobid, user))
            return [0, "Error during run of splitter: Please increase the filtering percentage! "
                       "Empty testset was produced!"]
        labels_filename = output_folder + 'training_labels.txt'

    # Copy the training labels to the output folder
    if split_dataset_flag != 1:
        try:
            copyfile(labels_filename, "{}training_labels.txt".format(output_folder))

            logging.info("PID:{}\tJOB:{}\tUSER:{}\tLabels copied successfully to output folder.".format(pid, jobid,
                                                                                                        user))
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during copying training labels to output"
                              " folder.".format(pid, jobid, user))
            return [0, "Error in copying the training labels."]
    # Attention: the modeller ts as input the 2d dataset array, and the labels list !
    try:
        # Select modeller
        select_modeller(data, labels, proteins, goal_significances, output_folder, selection_flag, population, generations, mutation_probability, arithmetic_crossover_probability, two_points_crossover_probability, num_of_folds, user, jobid, pid, thread_num)
    # print('after modeller')
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during run of modeller.".format(pid, jobid, user))
        return [0, "Exception during run of modeller: {}".format(e)]

    # Call plotting function
    try:
        draw_plots(output_folder + 'feature_selection' + os.sep + "AvgOfAvgGoalsPerGen.csv",
                  output_folder + 'average_performance.png')
        draw_plots(output_folder + 'feature_selection' + os.sep + "BestPerformancePerGen.csv",
                   output_folder + 'best_performance.png')
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during plotting.".format(pid, jobid, user))
        return [0, "Exception during plotting: {}".format(e)]

    # Get the names of the features and save them into a file
    try:

        markers = parsing_oneline(markers_filename)

        features = parsing_features(output_folder + 'feature_selection' + os.sep + "features_FinalFront1.csv")
        features = [int(x) for x in features]

        features_names = [markers[i] for i in features]
        write_oneline_to_newline_delimited_file(features_names, output_folder + "features_names.txt")
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during naming features.".format(pid, jobid, user))
        return [0, "Exception during naming features: {}".format(e)]


    # Copy the training labels to the output folder
    try:
        copyfile("{}feature_selection/features_FinalFront1.csv".format(output_folder),
                 "{}classification_models/features_selected.csv".format(output_folder))
        copyfile("{}feature_selection/number_of_svs.txt".format(output_folder),
                 "{}number_of_svs.txt".format(output_folder))
        copyfile("{}feature_selection/metrics.txt".format(output_folder),
                 "{}metrics.txt".format(output_folder))
        if selection_flag == 2 or selection_flag == 0:
            copyfile("{}feature_selection/cross_validation_roc_curve.png".format(output_folder),
                     "{}cross_validation_roc_curve.png".format(output_folder))
            copyfile("{}feature_selection/training_roc_curve.png".format(output_folder),
                     "{}training_roc_curve.png".format(output_folder))
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tNumber of SV/trees file and Metrics copied successfully to output "
                     "folder.".format(pid, jobid, user))

    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during copying number of SV/trees and Metrics to "
                          "output folder.".format(pid, jobid, user))
        return [0, "Error in copying the number of SV/trees and Metrics."]

    # Creating the selected features data from the selected features names and the initial dataset
    try:
        feature_names = parsing_oneline(output_folder + "features_names.txt")
        data = np.transpose(data)
        selected_biomarkers_data = [[] for _ in range(len(data))]
        for i in range(len(data)):
            for j in range(len(data[0])):
                if proteins[j] in feature_names:
                    selected_biomarkers_data[i].append(data[i][j])
        selected_biomarkers_data = np.transpose(selected_biomarkers_data)
        new_proteins = [x for x in proteins if x in feature_names]

    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException while producing selected features data".format(pid, jobid,
                                                                                                             user))
        return [0, "Exception while producing selected features data: {}".format(str(e))]

    # Creating the MQ files from the selected biomarkers dataset, the labels and the markers.
    try:
        if selection_flag == 0 or selection_flag == 2:
            labels = parsing_oneline(labels_filename)
            try:
                os.mkdir(output_folder + "Selected_MQ_files")
            except FileExistsError:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tSelected MQ files Directory already exists".format(pid,
                                                                                                               jobid,
                                                                                                               user))
            create_MQ_files(selected_biomarkers_data, new_proteins, labels, "Selected_MQ_",
                            "{0}Selected_MQ_files/".format(output_folder))

    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException while producing MQ files".format(pid, jobid, user))
        return [0, "Exception while producing MQ files: {}".format(str(e))]

    # Producing the differential expression files
    # Remember: assuming that the data are not logged !!!
    try:
        # unique_labels = list(set(labels))
        # Single label
        # if not isinstance(unique_labels[0], float) and len(unique_labels) == 2:
        if selection_flag == 2:
            try:
                os.mkdir(output_folder + "selected_biomarkers_files")
            except FileExistsError:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tWarning Selected Biomarkers files Directory already "
                                  "exists".format(pid, jobid, user))
            input_files = os.listdir(output_folder + "Selected_MQ_files/")
            proteins1, dataset1 = new_parse_data(output_folder + "Selected_MQ_files/" + input_files[0], "\t", user,
                                                 jobid, pid)
            proteins2, dataset2 = new_parse_data(output_folder + "Selected_MQ_files/" + input_files[1], "\t", user,
                                                 jobid, pid)

            create_selected_biomarkers_file(dataset1, dataset2, proteins1, logged_flag,
                                            output_folder + "selected_biomarkers_files/",
                                            "selected_biomarkers_file_{0}_VS_{1}.tsv".format(input_files[0][:-4],
                                                                                             input_files[1][:-4]))
        # Multi label
        # elif not isinstance(unique_labels[0], float) and len(unique_labels) > 2:
        elif selection_flag == 0:
            try:
                os.mkdir(output_folder + "selected_biomarkers_files")
            except FileExistsError:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tWarning Selected Biomarkers files Directory already "
                                  "exists".format(pid, jobid, user))
            input_files = os.listdir(output_folder + "Selected_MQ_files/")
            pairs_of_names = get_pairs_from_list(input_files)
            for pair in pairs_of_names:
                proteins1, dataset1 = new_parse_data(output_folder + "Selected_MQ_files/" + pair[0], "\t", user, jobid,
                                                     pid)
                proteins2, dataset2 = new_parse_data(output_folder + "Selected_MQ_files/" + pair[1], "\t", user, jobid,
                                                     pid)
                create_selected_biomarkers_file(dataset1, dataset2, proteins1, logged_flag,
                                                output_folder + "selected_biomarkers_files/",
                                                "selected_biomarkers_file_{0}_VS_{1}.tsv".format(pair[0][:-4],
                                                                                                 pair[1][:-4]))
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException while producing differential expression files".format(
            pid, jobid, user))
        return [0, "Exception while producing differential expression files: {}".format(str(e))]

    return [1, "Successful completion !"]


def transform_labels_to_numeric_twoclass(labels, unique_labels):
    """
    Transforms alphanumeric labels to numeric ones in the case of two classes.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the encoded 0-1 new labels
    """

    labels_dict = dict()
    labels_dict[unique_labels[0]] = 0
    labels_dict[unique_labels[1]] = 1
    # logging.info(str(labels_dict))
    new_labels = [labels_dict[x] for x in labels]
    # logging.info("Labels transformed to numeric successfully.")
    return new_labels


def transform_labels_to_alpha_twoclass(labels, unique_labels):
    """
    Transforms numeric labels back to alphanumeric according to given unique_labels one to one mapping.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the decoded labels with alphanumeric names
    """

    new_labels = []
    for x in labels:
        if x == 0:
            new_labels.append(unique_labels[0])
        elif x == 1:
            new_labels.append(unique_labels[1])
    # logging.info("Labels transformed to alphanumeric successfully.")
    return new_labels


def transform_labels_to_numeric_multiclass(labels, unique_labels):
    """
    Transforms alphanumeric labels to numeric ones in the case of multiple classes.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the encoded 0-1 new labels
    """

    labels_dict = dict()
    for i, label in enumerate(unique_labels):
        labels_dict[label] = i

    # logging.info(str(labels_dict))
    new_labels = [int(labels_dict[x]) for x in labels]
    # logging.info("Labels transformed to numeric successfully.")
    return new_labels


def transform_labels_to_alpha_multiclass(labels, unique_labels):
    """
    Transforms numeric labels back to alphanumeric according to given unique_labels one to one mapping.
    :param labels: the input labels
    :param unique_labels: the list with the two unique labels
    :return: new_labels: the decoded labels with alphanumeric names
    """

    labels_dict = dict()
    for i, label in enumerate(unique_labels):
        labels_dict[str(i)] = label

    # logging.info(str(labels_dict))
    new_labels = [labels_dict[str(x)] for x in labels]

    # logging.info("Labels transformed to alphanumeric successfully.")
    return new_labels


def find_delimiter_labels(filename):
    """
    Figures out which delimiter is being used in given file.
    :param filename: file to recognize delimiter
    :return: (string): "," if CSV content, "\t" if TSV content.
    """
    with open(filename, 'r') as file:
        line = next(file)
    if "\t" in line:
        return "\t"
    elif "," in line:
        return ","
    elif "," and "\t" in line:  # The case where the comma is the decimal separator (greek system)
        return "\t"


def parse_labels(labels_filename, data_length, selection_flag, user, jobid, pid):
    """
    Parse labels file according to each estimator selected
    :param labels_filename: filename of the labels file
    :param data_length: the length of the input data
    :param selection_flag: which estimator will be used, in order to know the type of labels
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return: labels list
    """

    delim_labels = find_delimiter_labels(labels_filename)

    # Parse input files
    labels = list()
    with open(labels_filename, "r") as labels_fid:
        for line in csv.reader(labels_fid, delimiter=delim_labels):
            for word in line:
                if selection_flag == 1:
                    labels.append(float(word.strip()))
                else:
                    labels.append(word.strip())
    label_length_diff = len(labels) - data_length

    if label_length_diff > 1:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tLabels file has {} more values than samples in dataset.".format(
            pid, jobid, user, label_length_diff))
        return [None, "Labels file has more values than samples in dataset"]
    elif label_length_diff < 0:
        logging.error("PID:{}\tJOB:{}\tUSER:{}\tLabels file has {} less values than samples in dataset.".format(
            pid, jobid, user, abs(label_length_diff)))
        return [None, "Labels file has less values than samples in dataset"]
    elif label_length_diff == 1:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tLabels file has a header".format(pid, jobid, user))
        labels = labels[1:]

    logging.info("PID:{}\tJOB:{}\tUSER:{}\tLabels file was successfully parsed!".format(pid, jobid, user))

    unique_labels = sorted(list(set(labels)))
    if selection_flag == 0 or selection_flag == 2:
        try:
            float(unique_labels[0])
            labels = [int(lbl) for lbl in labels]
        except ValueError:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tFound  non numeric labels.".format(pid, jobid, user))
            if selection_flag == 0:
                labels = transform_labels_to_numeric_multiclass(labels, unique_labels)
            else:
                labels = transform_labels_to_numeric_twoclass(labels, unique_labels)
        except Exception:
            logging.exception("PID:{}\tJOB:{}\tUSER:{}\tUnexpected Error.".format(pid, jobid, user))
            return [None, "Unexpected Error in reading labels file."]
    return [labels]


def checkdata(dataset, features, user, jobid, pid):
    """
    Check dataset for outliers etc.
    :param dataset: input dataset
    :param features: input features
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return: if dataset has outliers or not
    """
    '''pca = PCA(n_components=0.9, svd_solver='full')
    new_dataset = list()
    num_of_samples = 0
    for j in range(len(dataset[0])):
        new_dataset.append([])
        for i in range(len(dataset)):
            new_dataset[num_of_samples].append(float(dataset[i][j]))
        num_of_samples += 1
    dataset_new = pca.fit_transform(new_dataset)
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(dataset)
    '''
    variance = []
    for featline in dataset:
        variance.append(np.var(featline))
    feat_variance = np.var(variance)
    if feat_variance > 100:

        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tDataset has big variance {} between features min {} max {}.".format(
            pid, jobid, user, feat_variance, features[np.argmin(variance)], features[np.argmax(variance)]))
        return [True, 'Your dataset has big variance between features, please use InSyBio Biomarkers Preprocessing' \
                     ' tool or preprocess your data.']
    return [False, 'msg']


def main():
    dataset_filename = sys.argv[1]
    labels_filename = sys.argv[2]
    goal_significances_string = sys.argv[3]
    selected_comorbidities_string = sys.argv[4]
    selection_flag = int(sys.argv[5])
    split_dataset_flag = int(sys.argv[6])
    filtering_percentage = float(sys.argv[7])
    population = int(sys.argv[8])
    generations = int(sys.argv[9])
    mutation_probability = float(sys.argv[10])
    arithmetic_crossover_probability = float(sys.argv[11])
    two_points_crossover_probability = float(sys.argv[12])
    num_of_folds = int(sys.argv[13])
    filetype = int(sys.argv[14])
    has_feature_headers = int(sys.argv[15])
    has_sample_headers = int(sys.argv[16])
    output_folder = sys.argv[17]

    initLogging(selection_flag)
    result = run_model_and_splitter_selectors(
        dataset_filename, labels_filename, filtering_percentage, selection_flag, split_dataset_flag,
        goal_significances_string, selected_comorbidities_string, filetype, has_feature_headers, has_sample_headers,
        output_folder,0, population, generations, mutation_probability, arithmetic_crossover_probability,
        two_points_crossover_probability, num_of_folds)
    print(result)


if __name__ == "__main__":
    main()
