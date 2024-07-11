'''
get_features.py

A script that gets the features selected by the modeller out of the test set csv file and returns them as a csv file.

Input: input_testSet: the testSet.csv file from which the desired features will be extracted
       input_featureIDs: the featurelist.txt file from which the indexes of the desired features will be parsed
Output: output_testSet: the output_testSet.csv file where the desired features will be stored

Example of calling the code:
python3.6 get_features.py testSet.csv featurelist.txt output_testSet.csv
'''

import sys
import csv

def parsing_dataset(dataset_filename):
    '''
    Parses a csv dataset.

    Args:
        dataset_filename: the dataset filename

    Returns: 
        dataset: a list of lists
    '''
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
    return dataset

def saveDataToCSV(listOflists, outputfile):
    """ 
    Saves a list of lists to a CSV with tabs.

    Args:
        listOflists: a list of lists
        outputfile (string): the name of the output file    
    """
    with open(outputfile, "wt") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(listOflists)

def parsing_oneline(oneline_filename):
    '''
    Parses a file of one line containing integers

    Args:
        oneline_filename: a file with one line, with tab seperated numbers

    Returns:
        features: a list with  integers
    '''
    features = list()
    features_file = open(oneline_filename,'r')
    for line in features_file:
        word = line.split("\t")
        for w in word:
            features.append(int(w.rstrip()))
    return features


def create_testset_with_selected_features(input_testSet, input_featureIDs, output_testSet):
    '''
    Creates testset with selected features.

    Args:
        input_testSet: the input testSet file
        input_featureIDs: the file with the feature indexes
        output_testSet: the name of the output file
    '''
    # Parsing
    testSet = parsing_dataset(input_testSet)
    features = parsing_oneline(input_featureIDs)        

    # Keeping only the lines that correspond to the features we want
    output_testSet_list = list()
    for i in features:
        output_testSet_list.append(testSet[i])

    # Saving output testset
    saveDataToCSV(output_testSet_list, output_testSet)

def main():
    input_testSet = sys.argv[1]
    input_featureIDs = sys.argv[2]    
    output_testSet = sys.argv[3]

    create_testset_with_selected_features(input_testSet, input_featureIDs, output_testSet)    

if __name__ == "__main__":
    main()
