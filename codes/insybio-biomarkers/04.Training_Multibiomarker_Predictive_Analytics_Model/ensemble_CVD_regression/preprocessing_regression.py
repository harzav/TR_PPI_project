"""
preprocessing_regression.py
---------------------
Library with function that with given dataset and its Multiclass labels and from those it creates a test set of size 30%
(based on a given percentage) of the initial dataset and a training set of size 70% of the initial dataset. Also test
labels and training labels of similar sizes are being created. Also the training sets are returned as values from the
splitter_regression function.

"""

import sys
import numpy as np
from math import ceil
import logging


def parsing(dataset_filename, labels_filename, delimiter_dataset, delimiter_labels):
    """
    Parses the input dataset file and the labels file.

    Args:
        dataset_filename: the input dataset file
        labels_filename: the input labels file
        delimiter_dataset: (string) the delimiter of the dataset, "," or "\t"
        delimiter_labels: (string) the delimiter of the labels file, "," or "\t"

    Returns:
        dataset: (list of lists) the input data
        labels: (list) the input labels
    """
    dataset = list()
    with open(dataset_filename, "r") as dataset_fid:
        number_of_lines = 0
        for line1 in dataset_fid:
            dataset.append([])
            words = line1.split(delimiter_dataset)
            for i in range(len(words)):
                try:
                    dataset[number_of_lines].append(float(words[i]))
                except Exception:
                    dataset[number_of_lines].append(-1000)
            number_of_lines += 1
    # Dataset here is features X samples
    # print("Dataset file was successfully parsed!")  
    logging.info("Dataset file inside preprocessing was successfully parsed!")

    labels = list()
    with open(labels_filename, "r") as labels_fid:
        for line1 in labels_fid:
            words = line1.split(delimiter_labels)
            for i in range(len(words)):
                labels.append(float(words[i].strip()))
    # print("Labels file was successfully parsed!")
    logging.info("Labels file inside preprocessing was successfully parsed!")
    return dataset, labels


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


def find_delimiter_labels(labels_filename):
    """
    Figures out which delimiter is being used in given labels dataset.

    Args:
        labels_filename (string): the labels filename

    Returns:
        (string): "," if CSV content, "\t" if TSV content.
    """
    with open(labels_filename, 'r') as handle:
        head = next(handle)
    if "\t" in head:
        return "\t"
    elif "," in head:
        return ","
    elif "," and "\t" in head:  # The case where the comma is the decimal separator (greek system)
        return "\t"


def splitter_regression(dataset, labels, features, samples, filtering_percentage, output_folder, user, jobid, pid):
    """
    Splits the dataset into training and test set.

    Args:
        dataset (string): the input dataset
        labels (string): the labels filename
        features (list): dataset features
        samples (list): dataset sample names
        filtering_percentage (float): the percentage of the splitting; the proportion of the test set comparing to the
        whole dataset
        output_folder (string): output folder for produced files
        user: this job's username
        jobid: this job's ID in table biomarkers_jobs
        pid: this job's PID
    """
    # Transposing the initial dataset
    transposed = np.transpose(dataset)

    # Filtering
    # filtering_percentage = 0.3
    length_of_keep = ceil(filtering_percentage*len(labels))
    # print(length_of_keep)

    # Creating test data and test labels
    final_data = []
    for i in range(length_of_keep):        
        final_data.append(transposed[i])
    final_data_trans = np.transpose(final_data)    
        
    final_labels = []
    test_samples = []
    for i in range(length_of_keep):        
        final_labels.append(labels[i])
        test_samples.append(samples[i])

    # Writing test results to files
    try:
        with open(output_folder + "test_dataset.txt", "w") as output_file:
            for feature_line in final_data_trans:
                output_file.write("\t".join(str(values) for values in feature_line))
                output_file.write("\n")

        with open(output_folder + "test_dataset_with_headers.txt", "w") as output_file:
            output_file.write("Samples\t")
            output_file.write("\t".join(str(values) for values in test_samples))
            output_file.write("\n")
            for i, feature_line in enumerate(final_data_trans):
                output_file.write(features[i])
                output_file.write("\t")
                output_file.write("\t".join(str(values) for values in feature_line))
                output_file.write("\n")

        with open(output_folder + "test_labels.txt", "w") as labels_file:
            labels_file.write("\t".join(str(value) for value in final_labels))
    except Exception:
        logging.exception(
            "PID:{}\tJOB:{}\tUSER:{}\tException during splitting of dataset.".format(pid, jobid, user))
        return [False, "Exception during creating testing dataset.", ""]

    # Creating training data and training labels
    training_data = []
    for i in range(len(transposed)):
        if i > length_of_keep:
            training_data.append(transposed[i])
    training_data = np.transpose(training_data)

    training_labels = []
    training_samples = []
    for i in range(len(labels)):
        if i > length_of_keep:
            training_labels.append(labels[i])
            training_samples.append(samples[i])

    # Writing training results to files
    try:
        with open(output_folder + "training_dataset.txt", "w") as training_data_out:
            for feature_line in training_data:
                training_data_out.write("\t".join(str(values) for values in feature_line))
                training_data_out.write("\n")

        with open(output_folder + "training_dataset_with_headers.txt", "w") as output_file:
            output_file.write("Samples\t")
            output_file.write("\t".join(str(values) for values in test_samples))
            output_file.write("\n")
            for i, feature_line in enumerate(training_data):
                output_file.write(features[i])
                output_file.write("\t")
                output_file.write("\t".join(str(values) for values in feature_line))
                output_file.write("\n")

        with open(output_folder + "training_labels.txt", "w") as training_labels_out:
            training_labels_out.write("\t".join(str(value) for value in training_labels))
    except Exception:
        logging.exception(
            "PID:{}\tJOB:{}\tUSER:{}\tException during splitting of dataset.".format(pid, jobid, user))
        return [False, "Exception during creating training dataset.", ""]

    return [True, training_data, training_labels]


def main():	
    """dataset_filename = sys.argv[1]
    labels_filename = sys.argv[2]
    filtering_percentage = float(sys.argv[3])
    output_folder = sys.argv[4]    
    splitter_regression(dataset_filename, labels_filename, filtering_percentage, output_folder)"""


if __name__ == "__main__":
    main()

