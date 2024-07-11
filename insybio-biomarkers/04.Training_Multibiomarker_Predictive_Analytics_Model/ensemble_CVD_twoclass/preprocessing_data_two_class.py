"""
preprocessing_data_two_class.py
---------------------
Library with function that with given dataset and its Multiclass labels and from those it creates a test set of size 30%
(based on a given percentage) of the initial dataset and a training set of size 70% of the initial dataset. Also test
labels and training labels of similar sizes are being created. Also the training sets are returned as values from the
splitter_twoclass function.

"""

import sys
import numpy as np
from math import ceil
import copy
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
                labels.append(words[i].strip())
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


def splitter_twoclass(dataset, labels, labelnames, features, samples, filtering_percentage, output_folder, user, jobid,
                      pid):
    """
    Splits the dataset into training and test set.

    Args:
        dataset (list): the input dataset
        labels (list): the dataset labels
        features (list): dataset features
        samples (list): dataset sample names
        filtering_percentage (float): the percentage of the splitting; the proportion of the test set comparing to the
         whole dataset
        output_folder (string): path to output folder
        user: this job's username
        jobid: this job's ID in table biomarkers_jobs
        pid: this job's PID
    """

    # Transpose dataset so that we have the samples as rows and the biomarkers as columns
    transposed = np.transpose(dataset)

    # Creating a set of the unique labels
    unique_labels = list(set(labels))

    # Initializing the dictionary with keys the names of the labels and values to zero
    indexes = {str(x): 0 for x in unique_labels}
    old_indexes = copy.deepcopy(indexes)

    # Calculating how many labels of each kind we have
    for x in labels:
        for index in unique_labels:
            if x == index:
                indexes[str(x)] += 1

    # Creating a dictionary with the number of the filtered labels from each kind
    # filtering_percentage = 0.3
    new_indexes = {key: ceil(filtering_percentage*val) for (key, val) in indexes.items()}

    # Creating the keep_list which has zeros and ones and shows which labels or data we'll keep
    keep_list = []
    for x in labels:
        for index in unique_labels:
            if x == index:
                old_indexes[str(x)] += 1
                if old_indexes[str(x)] < new_indexes[str(x)]:
                    keep_list.append(1)
                else:
                    keep_list.append(0)

    # Appending test data to temporary lists

    final_data = []
    for i in range(len(transposed)):
        if keep_list[i] == 1:
            final_data.append(transposed[i])

    final_labels = []
    test_samples = []
    for i in range(len(labelnames)):
        if keep_list[i] == 1:
            final_labels.append(labelnames[i])
            test_samples.append(samples[i])

    final_data_trans = np.transpose(final_data)

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
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during splitting of dataset.".format(pid, jobid, user))
        return [False, "Exception during creating testing dataset.", ""]

    # Appending training data to temporary lists
    final_training_data = []
    for i in range(len(transposed)):
        if keep_list[i] == 0:
            final_training_data.append(transposed[i])
    final_training_data = np.transpose(final_training_data)

    final_training_labels = []
    output_training_labels = []
    training_samples = []
    for i in range(len(labelnames)):
        if keep_list[i] == 0:
            final_training_labels.append(labelnames[i])
            output_training_labels.append(labels[i])
            training_samples.append(samples[i])

    # Writing training results to files
    try:
        with open(output_folder + "training_dataset.txt", "w") as training_data_out:
            for feature_line in final_training_data:
                training_data_out.write("\t".join(str(values) for values in feature_line))
                training_data_out.write("\n")

        with open(output_folder + "training_dataset_with_headers.txt", "w") as output_file:
            output_file.write("Samples\t")
            output_file.write("\t".join(str(values) for values in test_samples))
            output_file.write("\n")
            for i, feature_line in enumerate(final_training_data):
                output_file.write(features[i])
                output_file.write("\t")
                output_file.write("\t".join(str(values) for values in feature_line))
                output_file.write("\n")

        with open(output_folder + "training_labels.txt", "w") as training_labels_out:
            training_labels_out.write("\t".join(str(value) for value in final_training_labels))
    except Exception:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during splitting of dataset.".format(pid, jobid, user))
        return [False, "Exception during creating training dataset.", ""]

    return [True, final_training_data, output_training_labels]


def main():
    """dataset_filename = sys.argv[1]
    labels_filename = sys.argv[2]
    filtering_percentage = float(sys.argv[3])
    output_folder = sys.argv[4]
    splitter_twoclass(dataset_filename, labels_filename, filtering_percentage, output_folder)"""


if __name__ == "__main__":
    main()
