"""
Unused libraries

import time
from numpy import matrix
import plotly.plotly as py
import plotly.tools as tls
import plotly
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import mstats
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from itertools import compress
from scipy import stats, linalg
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import calinski_harabaz_scoreos.environ['R_HOME'] = "/usr/lib/R/"
os.environ['R_USER'] = "/usr/lib/python3/dist-packages/rpy2/"
from sklearn.preprocessing import Imputer

#import rpy2.robjects.lib.limma as limmaos.environ['R_HOME'] = "/usr/lib/R/"
os.environ['R_USER'] = "/usr/lib/python3/dist-packages/rpy2/"

"""
import logging
import os
import statistics
import numpy as np
import pandas as pd
import scipy.stats as st
import math
import sys
import csv
import matplotlib
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smm
import copy
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

os.environ['R_HOME'] = "/usr/lib/R/"
os.environ['R_USER'] = "/usr/lib/python3/dist-packages/rpy2/"
from rpy2 import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

ro.numpy2ri.activate()
from rpy2.robjects import r
import rpy2.robjects.lib.ggplot2 as ggplot2

gplots = importr("ggplot2")  # change gplots = importr("gplots")
# gplots = importr("gplots")
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

limma = importr('limma')
# statmod=importr('statmod')
ggrepel = importr('ggrepel')
lattice = importr('lattice')
beanplot = importr('beanplot')


# from sklearn.preprocessing import Imputer


def merge_lists(data, position, list_to_insert, insert_as_row, insert_as_col):
    """
    Helper of insert_column_between_data() function to merge the data
    :param data:
    :param position:
    :param list_to_insert:
    :param insert_as_row:
    :param insert_as_col:
    :return:
    """
    if insert_as_row:
        data.insert(position, list_to_insert)

    elif insert_as_col and (len(list_to_insert) == len(data)):
        for i in range(len(data)):
            data[i].insert(position, list_to_insert[i])
        logging.debug("List inserted in column {} of previous data", position)
    else:
        logging.debug("Wrong dimensions given \n")
        if len(list_to_insert) > len(data):
            logging.debug("length of list to be inserted is bigger than the length of input data")
        else:
            logging.debug("length of list to be inserted is smaller than the length of input data")
        logging.debug("Previous data are returned")

    return data


def insert_column_between_data(data, list_to_insert, position, col_name="", insert_as_row=1, insert_as_col=0):
    """
    Function that inserts a new column to already existing dataframe

    :param data: a pandas dataframe, a list of lists or a list of tuples supported
    :param list_to_insert: the list with same size of dataframe to be inserted into dataframe
    :param position: the position where it should be inserted
    :param col_name: the name the column should have
    :param insert_as_row default is 1, choice to insert the input list as row
    :param insert_as_col default is 0, choice to insert the input list as column
    :return: the new dataframe
    """

    if isinstance(data, pd.DataFrame):  # dataframe
        logging.debug('data are dataframe')
        insert_list = pd.DataFrame(list_to_insert)
        data.insert(position, col_name, insert_list)
    elif all(isinstance(d, list) for d in data):  # list of lists
        logging.debug("list of lists")
        data = merge_lists(data, position, list_to_insert, insert_as_row, insert_as_col)
    elif all(isinstance(d, tuple) for d in data):  # list of tuples
        logging.debug("list of tuples")
        data = [list(d) for d in data]  # transform data to list of lists
        data = merge_lists(data, position, list_to_insert, insert_as_row, insert_as_col)
    elif type(data).__module__ == np.__name__:  # numpy nd array
        logging.debug('numpy nd.array')
        data = [list(d) for d in data]  # transform data to list of lists
        data = merge_lists(data, position, list_to_insert, insert_as_row, insert_as_col)
    else:
        logging.debug('unkown input type')
    return data


def parse_data(data_filename):
    num_of_lines = 0
    proteins = list()
    data = list()
    samples = list()
    with open(data_filename) as data_fname:
        for line in csv.reader(data_fname, delimiter="\t"):
            if num_of_lines == 0:
                for j in range(len(line)):
                    if j > 0:
                        samples.append(line[j].strip())
            else:
                proteins.append(line[0])
                data.append([])
                for j in range(len(line)):
                    if j > 0:
                        # print(num_of_lines)
                        # print(j)
                        if line[j] != '':
                            data[num_of_lines - 1].append(float(line[j]))
                        else:
                            data[num_of_lines - 1].append('')
            num_of_lines += 1
    logging.debug('Data were successfully parsed!')
    return [proteins, data, samples]


def parse_commorbidities(samples, commorbidities_filename, commorbidities_types):
    # age=list()
    # sex=list()
    # statin=list()
    # a_or_b=list()
    # commorbidities=dict()
    commorbidities_flag = 0
    # num_of_lines=0
    commorbidities = list()
    with open(commorbidities_filename) as commorbidities_fname:
        for line in csv.reader(commorbidities_fname, delimiter="\t"):
            commorbidities_flag = 1
            if len(commorbidities) == 0:
                for i in range(len(line)):
                    commorbidities.append([])
            for i in range(len(line)):
                if commorbidities_types[i] == "1" or commorbidities_types[i] == 1:
                    commorbidities[i].append(int(math.floor(float(line[i].strip()))))
                else:
                    commorbidities[i].append(line[i].strip())
    return commorbidities, commorbidities_flag


def create_differential_expression_file(data1: list, data2: list, gene_names: list, logged_flag: int,
                                        sign_threshold: float, method: int,
                                        output_folder: str, filename_result: str):
    """
    Creates the differential expression file by reading data from two MQ files

    Args:
        data1 (list): the first input dataset
        data2 (list): the second input dataset
        gene_names (list): the gene names
        logged_flag (int): 0 for not logarithmic, 1 for logarithmic
        sign_threshold (float): the pvalue threshold (significance threshold)
        method (integer): 1 for parametric, 2 for non parametric test
        output_folder (string): the output folder
        filename_result (string): the name of the output file

    Returns:
        nothing, prints data to file named filename_result.
    """
    p_value = list()
    avgs1 = list()
    avgs2 = list()
    fold_changes = list()
    # data1_new = list()s
    print(len(data1))
    for i in range(0, len(data1) - 1):  # maybe -1 needed
        # for j in (data1[i]):
        #     if j != "":
        #         data1_new.append(j)
        # print(len(data1_new))
        # print(sum(data1_new))
        avgs1.append(float(sum(data1[i])) / max(len(data1[i]), 1))
        avgs2.append(float(sum(data2[i])) / max(len(data2[i]), 1))
        if logged_flag == 0:
            if avgs1[i] == 0:
                fold_changes.append('')
            else:
                fold_changes.append(avgs2[i] / avgs1[i])
        else:
            fold_changes.append(avgs2[i] - avgs1[i])

        if method == 1:
            value = st.ttest_ind(data1[i], data2[i])
        else:
            value = st.ranksums(data1[i], data2[i])

        p_value.append(value[1])
    p_value_rank = st.rankdata(p_value, method='ordinal')

    for i in range(0, len(p_value_rank)):
        p_value_rank[i] = int(p_value_rank[i])

    p_value_rank = p_value_rank.tolist()
    output_file = open(output_folder + filename_result, "w")
    header = "Gene name\tPvalue\tAvg1\tAvg2\tFold changes\n"
    output_file.write(header)
    for i in range(0, len(p_value_rank)):
        value = p_value[p_value_rank.index(i + 1)]
        if value < sign_threshold:
            output_file.write(gene_names[p_value_rank.index(i + 1)] + "\t"
                              + str(p_value[p_value_rank.index(i + 1)]) + "\t"
                              + str(avgs1[p_value_rank.index(i + 1)]) + "\t"
                              + str(avgs2[p_value_rank.index(i + 1)]) + "\t"
                              + str(fold_changes[p_value_rank.index(i + 1)]) + "\n")
        else:
            break
    output_file.close()


def new_parse_data(data_filename, delimiter):
    """
    Parses data.

    Args:
        data_filename (string): dataset filename
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
    # print('Data were successfully parsed!')
    logging.info('Data were successfully parsed!')
    return [proteins, data]


def print_data_nosamples(data, markers, folder_name, filename):
    """
    Writes data and labels to a file.

    Args:
        data: input data (list of lists)
        markers: input biomarkers (list)
        folder_name: output folder
        filename: output filename

    Returns: doesn't return anything, only writes markers, sample names and data to a file.
    """
    file = open(folder_name + filename, 'w')
    message = ''
    for i in range(len(data)):
        message += markers[i]
        for j in range(len(data[0])):
            message += '\t' + str(data[i][j])
        message += '\n'
    file.write(message)
    file.close()


def print_data2(data, markers, labels, folder_name, filename):
    """
    Writes data and labels to a file.

    Args:
        data: input data (list of lists)
        markers: input biomarkers (list)
        labels: input labels, sample names (list)
        folder_name: output folder
        filename: output filename

    Returns: doesn't return anything, only writes markers, sample names and data to a file.
    """
    with open(folder_name + filename, 'w') as file:
        message = ''
        for i in range(len(data[0])):
            message += '\t' + labels[i]
        message += '\n'
        for i in range(len(data)):
            message += markers[i]
            for j in range(len(data[0])):
                message += '\t' + str(data[i][j])
            message += '\n'
        file.write(message)


def print_significant_data_better(data, markers1, markers2, samples, pvals, p_value_threshold, output_folder):
    """
    Subsets the input data and keeps only the data that correspond to the significant molecules (markers).

    Args:
        data (list of lists): input data
        markers1 (list): diff_proteins or markers, depending on the method (parametric or not parametric)
        markers2 (list): the original list of markers
        samples (list) the samples names list
        pvals (list): the p-Values list
        p_value_threshold (float): the filtering coefficient for the p-Value
        output_folder (string): the output folder

    Returns: prints data to a file
    """
    new_proteins = list()
    new_data = list()
    for j in range(len(markers1)):
        if pvals[j] < float(p_value_threshold):
            new_proteins.append(markers1[j])

    indexes = [markers2.index(new_proteins[i]) for i in range(len(new_proteins))]
    for i in range(len(indexes)):
        new_data.append(data[i])
    # new_data = [data[i] for i in indexes]
    if len(new_data) == 0:
        return [True, "The list new_data is empty, problem with pvalue threshold."]
    else:
        print_data2(new_data, new_proteins, samples, output_folder, "significant_molecules_dataset.tsv")
        return [False, '']


def create_MQ_files(dataset, markers, labels, output_filename, output_folder):
    """
    Creates the MQ files of given dataset, samples, markers and labels.

    Args:
         dataset (list): list of lists of input data, with markers as rows and samples as columns
         markers (list): list of marker names
         labels (list): list of input labels
         output_filename (string): the prefix of the name of the output file
         output_folder (string): the name of the output folder

    Returns:
        nothing: prints data to file
    """
    data_transp = np.transpose(dataset)
    unique_labels = list(set(labels))
    for tag in unique_labels:
        new_dataset = list()
        # new_samples = list()
        for i, label in enumerate(labels):
            if tag == label:
                new_dataset.append(data_transp[i])
            # new_samples.append(samples[i])
        new_dataset = np.transpose(new_dataset)
        print_data_nosamples(new_dataset, markers, output_folder, output_filename + tag + ".tsv")


def differential_expression_analysis_new(control, condition, proteins, data, labels, comorbidities_flag,
                                         comorbidities, comorbidities_type, tags, folder_name, suffix):
    """
    Parametric testing using R limma
    :param control: Control dataset
    :param condition: COndition dataset
    :param proteins: list of features
    :param data: Whole dataset
    :param labels: labels list
    :param comorbidities_flag: if there are commorbidities
    :param comorbidities: list of commorbidities
    :param comorbidities_type: types of commorbidiies
    :param tags: unique labels
    :param folder_name: output folder
    :param suffix: suffix for output files
    :return:
    """

    # def differential_expression_analysis_new(proteins, data, labels, comorbidities_flag,
    # comorbidities,comorbidities_type, tags,folder_name,suffix): #change myList -> tags
    proteins_vector = ro.StrVector(proteins)  # all proteins in an one dimensional list

    logging.debug(tags)

    data_array = np.asarray(data)

    data_robject = ro.r.matrix(data_array, nrow=len(data_array))
    data_robject.rownames = proteins_vector  # all proteins
    labels_array = np.asarray(labels)  # all labels from file
    labels_robject = ro.r.matrix(labels_array, nrow=len(labels_array))  # transform to R object
    # print(labels_robject)
    # design=model.matrix(ro.FactorVector(labels_robject))
    # print(design)
    ro.r.assign('data', data_robject)
    ro.r.assign('labels', labels_robject)

    if comorbidities_flag == 0:
        ro.r('design<-model.matrix(~factor(labels))')
    else:
        command = 'design<-model.matrix(~factor(labels)'
        for i in range(len(comorbidities)):
            ro.r.assign('comorbidity_' + str(i), ro.r.matrix(np.asarray(comorbidities[i]), nrow=len(labels_array)))
            if comorbidities_type[i] == '0':
                command += '+factor(comorbidity_' + str(i) + ')'
            else:
                command += '+comorbidity_' + str(i)
        command += ')'
        ro.r(command)

    # print(ro.r('dim(data)'))
    # print(ro.r('dim(design)'))
    ro.r('fit <-lmFit (data, design)')
    ro.r('fit2<-eBayes(fit)')

    # change
    ro.r("top<-topTable(fit2,coef=2, number=length(data))")
    table = np.asarray(ro.r("top"))
    columns = list(ro.r("colnames(top)"))  # logFC	AveExpr		t	P.Value		adj.P.Val	B
    proteins_ids = list(ro.r("rownames(top)"))  # all the proteins
    table_array = (np.asarray(table))

    average_control = list()
    average_condition = list()

    for i in range(len(proteins)):
        control_data_per_protein = list()
        condition_data_per_protein = list()
        for j in range(len(control[i])):
            control_data_per_protein.append(control[i][j])
        for j in range(len(condition[i])):
            condition_data_per_protein.append(condition[i][j])

        # change
        # creating the 2 lists of averages of control and condition data
        average_control.append(statistics.mean(control_data_per_protein))  # CI.L
        average_condition.append(statistics.mean(condition_data_per_protein))  # CI.R

    # change insert into column names and data
    columns.insert(1, "CI.L")
    columns.insert(2, "CI.R")

    # change insert into table the lists of average control and condition
    table_array = insert_column_between_data(table_array, average_control, 1, col_name="CI.L", insert_as_row=0,
                                             insert_as_col=1)
    table_array = insert_column_between_data(table_array, average_condition, 2, col_name="CI.R", insert_as_row=0,
                                             insert_as_col=1)

    with open(folder_name + 'diff_exp_results' + suffix + '.txt', 'w') as output_file:
        output_file.write('\t')
        for title in columns:
            output_file.write(str(title) + '\t')
        output_file.write('\n')

        for i in range(len(table_array)):  # 309 rows
            output_file.write(str(proteins_ids[i]) + '\t')
            for j in range(len(columns)):  # 8 columns
                output_file.write(str(table_array[i][j]) + '\t')
            output_file.write('\n')

    if not os.path.isdir(folder_name + "top20/"):
        os.mkdir(folder_name + "top20/")
    header = 'IDs\tInitial Pvalue\tAdjusted Pvalue\tFold Change\tAverage Expression\n'
    with open(folder_name + "top20/" + "all_pvals_top20.txt", 'w') as output_file:
        '''output_file.write('\t')
        for i, title in enumerate(columns):
            output_file.write(str(title))
            if i < len(columns) - 1:
                output_file.write('\t')'''
        output_file.write(header)
        for i in range(len(table_array)):
            if i < 20:
                output_file.write(str(proteins_ids[i]) + '\t')
                output_file.write(str(table_array[i][5]) + '\t' + str(table_array[i][6]) + '\t' +
                                  str(table_array[i][0]) + '\t' + str(table_array[i][1]) + '\n')
                '''for j in range(len(columns)):
                    output_file.write(str(table_array[i][j]))
                    if j < len(table_array) - 1:
                        output_file.write('\t')
                output_file.write('\n')'''

    logging.debug("differential analysis successfully done!")
    return [table, proteins_ids, columns]


def print_volcano_plots(proteins, pvals, log_fold_changes, filename, filename2, p_value_threshold, volcano_width,
                        volcano_height, volcano_titles, volcano_axis_labels, volcano_labels, volcano_axis_relevance,
                        volcano_criteria, abs_log_fold_changes_threshold, volcano_labeled):
    """
    Print The selected Volcanoplots according to the input parameters
    :param proteins: proteins
    :param pvals: p values
    :param log_fold_changes: log fold changes
    :param filename: labeled file
    :param filename2: unlabeled file
    :param p_value_threshold: the p value threshold
    :param volcano_width: width of png
    :param volcano_height: height of png
    :param volcano_titles: size of titles
    :param volcano_axis_labels: size of axis labels
    :param volcano_axis_relevance: the relevance of axis y to axis x , default =1
    :param volcano_labels: size of labels inside the volcano plot
    :param abs_log_fold_changes_threshold: absolute log fold changes threshold, default =0
    :param volcano_criteria: 1 -> all, 2 -> non-overlapping , 3 -> p-value threshold and absolute log fold changes
    threshold
    :param volcano_labeled: 0 -> unlabeled, 1 -> unlabeled, 2 -> both
    :return: None, creates the volcano plots
    """
    thresholds = list()
    selected_proteins = list()
    absolute_log_fold_changes = list()

    for i in range(len(log_fold_changes)):
        absolute_log_fold_changes.append(abs(log_fold_changes[i]))

    if volcano_criteria == 1:  # all == all proteins without any threshold/s
        logging.debug("non criteria")
        for i in range(len(pvals)):
            # if pvals[i] < 1: # practically  all the proteins
            thresholds.append(1)
            selected_proteins.append(proteins[i])
        overlap = "FALSE"

    elif volcano_criteria == 2:
        logging.debug("overlapping criteria")
        for i in range(len(pvals)):
            # if pvals[i] < 1: # practically  all the proteins
            thresholds.append(1)
            selected_proteins.append(proteins[i])
        overlap = "TRUE"

    elif volcano_criteria == 3:
        if not abs_log_fold_changes_threshold == 0 or p_value_threshold == 1:  # if not chosen to plot all
            logging.debug("p value threshold and absolute log fold changes criteria")
            for i in range(len(pvals)):
                if pvals[i] < p_value_threshold and absolute_log_fold_changes[i] > abs_log_fold_changes_threshold:
                    thresholds.append(1)
                    selected_proteins.append(proteins[i])
                else:
                    thresholds.append(0)
                    selected_proteins.append('')
        else:
            logging.debug("non criteria")
            for i in range(len(pvals)):
                thresholds.append(1)
                selected_proteins.append(proteins[i])
        overlap = "FALSE"
    ro.FactorVector(list(map(str, thresholds)))  # ro.FactorVector(thresholds)
    log_pvals = list()
    colors = list()
    for i in range(len(pvals)):
        if pvals[i] < 0.001:
            colors.append('red2')
        elif pvals[i] < 0.01:
            colors.append('orange1')
        elif pvals[i] < 0.05:
            colors.append('darkgreen')
        else:
            colors.append('darkblue')

        if pvals[i] == 0:
            log_pvals.append(250)
        else:
            log_pvals.append(-1 * math.log10(abs(pvals[i])))  # change log_pvals.append(-1*math.log10(pvals[i]))

    limit1 = -1 * max(absolute_log_fold_changes) - 0.5
    limit2 = max(absolute_log_fold_changes) + 0.5

    colormap_raw = [['red2', '#ff0000'], ['orange1', '#FFA500'], ['darkgreen', '#006400'], ['darkblue', '#003366']]
    colormap_labels = [['red2', 'P < 0.001'], ['orange1', 'P < 0.01'], ['darkgreen', 'P < 0.05'],
                       ['darkblue', 'P > 0.05']]

    colormap = ro.StrVector([elt[1] for elt in colormap_raw])
    colormap.names = ro.StrVector([elt[0] for elt in colormap_raw])
    ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
    ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')

    df_dict = {'Ids': ro.StrVector(selected_proteins), 'threshold': ro.IntVector(thresholds),
               'log2FC': ro.FloatVector(log_fold_changes), 'MinusLog10Pvals': ro.FloatVector(log_pvals),
               'colors': ro.StrVector(colors)}

    df = ro.DataFrame(df_dict)
    # labeled plot
    # axis.text.y -> size of axis y numbers-scale
    # axis.text.x -> size of axis x numbers-scale
    # text -> size of letters of axis titles
    # ggrepel. size -> size of labels
    if max(log_pvals) == min(log_pvals):
        volcano_axis_relevance = (limit2 - limit1) / (max(log_pvals) - min(log_pvals) + 0.1) * volcano_axis_relevance
    else:
        volcano_axis_relevance = ((limit2 - limit1)/(max(log_pvals) - min(log_pvals))) * volcano_axis_relevance
    if volcano_labeled == 0:  # unlabeled

        r.png(filename2, width=volcano_width, height=volcano_height, units='cm', res=600)  # change width=7, height=6

        ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
        ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')
        gp = ggplot2.ggplot(data=df) + \
             ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids', colour='colors') + \
             ggplot2.xlim(limit1, limit2) + ggplot2.geom_point(size=0.8) + \
             ggplot2.scale_colour_manual("", values=colormap, breaks=colormap.names,
                                         labels=[elt[1] for elt in colormap_labels]) + \
             ggplot2.theme(**{'text': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.x': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.y': ggplot2.element_text(size=volcano_titles, angle=90)}) + \
             ggplot2.theme(**{'axis.text.x': ggplot2.element_text(size=volcano_axis_labels)}) + \
             ggplot2.theme(**{'axis.text.y': ggplot2.element_text(size=volcano_axis_labels, angle=90)}) + \
             ggplot2.coord_fixed(ratio=volcano_axis_relevance)
        gp.plot()

    elif volcano_labeled == 1:  # labeled

        r.png(filename, width=volcano_width, height=volcano_height, units='cm', res=600)  # change width=7, height=6

        gp = ggplot2.ggplot(data=df) + \
             ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids', colour='colors') + \
             ggplot2.xlim(limit1, limit2) + ggplot2.geom_point(size=0.8) + \
             ggrepel.geom_text_repel(colour='black', check_overlap=overlap, size=volcano_labels, force=2, max_time=1) + \
             ggplot2.scale_colour_manual("", values=colormap, breaks=colormap.names,
                                         labels=[elt[1] for elt in colormap_labels]) + \
             ggplot2.theme(**{'text': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.x': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.y': ggplot2.element_text(size=volcano_titles, angle=90)}) + \
             ggplot2.theme(**{'axis.text.x': ggplot2.element_text(size=volcano_axis_labels)}) + \
             ggplot2.theme(**{'axis.text.y': ggplot2.element_text(size=volcano_axis_labels, angle=90)}) + \
             ggplot2.coord_fixed(ratio=volcano_axis_relevance)
        gp.plot()

    elif volcano_labeled == 2:  # both labeled and unlabeled

        r.png(filename2, width=volcano_width, height=volcano_height, units='cm', res=600)  # change width=7, height=6
        # unlabeled
        gp = ggplot2.ggplot(data=df) +  \
             ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids', colour='colors') + \
             ggplot2.xlim(limit1, limit2) + ggplot2.geom_point(size=0.8) + \
             ggplot2.scale_colour_manual("", values=colormap, breaks=colormap.names,
                                         labels=[elt[1] for elt in colormap_labels]) + \
             ggplot2.theme(**{'text': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.x': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.y': ggplot2.element_text(size=volcano_titles, angle=90)}) + \
             ggplot2.theme(**{'axis.text.x': ggplot2.element_text(size=volcano_axis_labels)}) + \
             ggplot2.theme(**{'axis.text.y': ggplot2.element_text(size=volcano_axis_labels, angle=90)}) + \
             ggplot2.coord_fixed(ratio=volcano_axis_relevance)
        #gp = gp + ggplot2.labs(y="-log10(Pval)")
        #print(gp)
        #dev.off()
        gp.plot()
        # unlabeled plot
        ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
        ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')

        r.png(filename, width=volcano_width, height=volcano_height, units='cm', res=600)  # change width=7, height=6
        gp = ggplot2.ggplot(data=df) + \
             ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids', colour='colors') + \
             ggplot2.xlim(limit1, limit2) + ggplot2.geom_point(size=0.8) + \
             ggrepel.geom_text_repel(colour='black', check_overlap=overlap, size=volcano_labels, force=2, max_time=1) + \
             ggplot2.scale_colour_manual("", values=colormap, breaks=colormap.names,
                                         labels=[elt[1] for elt in colormap_labels]) + \
             ggplot2.theme(**{'text': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.x': ggplot2.element_text(size=volcano_titles)}) + \
             ggplot2.theme(**{'axis.title.y': ggplot2.element_text(size=volcano_titles, angle=90)}) + \
             ggplot2.theme(**{'axis.text.x': ggplot2.element_text(size=volcano_axis_labels)}) + \
             ggplot2.theme(**{'axis.text.y': ggplot2.element_text(size=volcano_axis_labels, angle=90)}) + \
             ggplot2.coord_fixed(ratio=volcano_axis_relevance)
        #gp = gp + ggplot2.labs(y="-log10(Pval)")
        #print(gp)
        #dev.off()
        gp.plot()

    r("while (!is.null(dev.list()))  dev.off()")
    # r("dev.off()")
    logging.debug("Volcano plots were successfully created")


def print_beanplots(new_proteins, proteins, data, values_list, labels_1, folder, pvals, p_value_threshold, fold_changes,
                    beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles,
                    beanplot_axis_titles):
    """
    Print Beanplots for each pair of Labels for each feature in the dataset
    :param new_proteins: list of dataset features
    :param proteins: list of dataset features
    :param data: input 2d list with dataset
    :param values_list: Unique labels
    :param labels_1: list of labels
    :param folder: folder for output files
    :param pvals: list of calculated pvalues
    :param p_value_threshold: threshold for filtering p values
    :param fold_changes: list of fold change between samples
    :param beanplot_width: width of beanplot image
    :param beanplot_height: height of beanplot image
    :param beanplot_axis: scaling of axis of axis labels, 1=default, 1.5 is 50% larger, 0.5 is 50%
    :param beanplot_xaxis: scaling of axis of xaxis labels, 1=default, 1.5 is 50% larger, 0.5 is 50%
    :param beanplot_yaxis: scaling of axis of yaxis labels, 1=default, 1.5 is 50% larger, 0.5 is 50%
    :param beanplot_titles: scaling of of titles, 1=default, 1.5 is 50% larger, 0.5 is 50%
    :param beanplot_axis_titles: scaling of axis titles, 1=default, 1.5 is 50% larger, 0.5 is 50%
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    for j in range(len(proteins)):
        # logging.info(proteins[j])
        num_of_cat = list()
        num_of_cat.append(0)
        num_of_cat.append(0)
        if pvals[j] < p_value_threshold:
            control_table = list()

            for k in range(len(labels_1)):

                if labels_1[k] == values_list[0]:

                    ind = new_proteins.index(proteins[j])
                    if data[ind][k] != '' and data[ind][k] != -1000 and not np.isnan(data[ind][k]):
                        num_of_cat[0] += 1
                        control_table.append(float(data[ind][k]))

            case_table = list()

            for k in range(len(labels_1)):

                if labels_1[k] == values_list[1]:

                    ind = new_proteins.index(proteins[j])
                    if data[ind][k] != '' and data[ind][k] != -1000 and not np.isnan(data[ind][k]):
                        num_of_cat[1] += 1
                        case_table.append(float(data[ind][k]))
            protein_name = replace_strange_chars(proteins[j])
            #selected_features = ['RBMS3_RBMS3_RBMS3_RBMS3_RBMS3_RBMS3','CYP3A7','ASPRV1_ASPRV1','CCNI2','CD151_CD151_CD151_CD151','BUB3_BUB3_BUB3_BUB3','FKBP1AP1','SETD4','LENG9','ZNF416_ZNF416']
            #if protein_name not in selected_features:
            #    continue
            try:
                # ro.r.png(folder + protein_name + '_beanplot.png', width=beanplot_width, height=beanplot_height,
                #          units='cm', res=1000)  # change: sizes added
                ro.r.png(folder + protein_name + '_beanplot.png', width=8, height=8,
                         units='cm', res=600)  # change: sizes added
            except rpy2.rinterface.RRuntimeError:
                logging.info("Protein: {}, name:{}".format(proteins[j], protein_name))
                raise rpy2.rinterface.RRuntimeError

            ro.r.assign("control", ro.r.matrix(np.asarray(control_table), nrow=len(control_table)))
            ro.r.assign("case", ro.r.matrix(np.asarray(case_table), nrow=len(case_table)))
            # ro.r('library("beanplot")')

            new_list = list()
            for i in range(len(values_list)):
                # new_list.append(' \n \n \n\n'+values_list[i]+'\n(n=' +str(num_of_cat[i])+')')
                new_list.append(values_list[i])  # new_list.append('  \n'+values_list[i])
            logging.info("labels of comparison {}".format(values_list))

            ro.r.assign('labels_of_comparison', ro.StrVector(values_list))
            # print(ro.FactorVector(values_list))
            # print(ro.StrVector(values_list))

            ro.r("par(las=1,cex.axis=1,cex.lab=1, font.lab=1)")
            try:
                ro.r('beanplot(control, case, bw="nrd0", log = "", names=labels_of_comparison)')
            # ro.r('beanplot(control, bw="nrd0", log = "")')
            except rpy2.rinterface.RRuntimeError:
                logging.info(case_table)

            # ro.r('title("' + str(proteins[j]) + '\nLog2(FC)=' + str(round(fold_changes[j], 4)) +
            #      '",xlab="Adj. P-value: ' + str(round(pvals[j], 4)) + '", ylab="Log2(RQ)", cex.main=' +
            #      str(beanplot_titles) + ', cex.axis=' + str(beanplot_axis_titles) + ', cex.labs=' +
            #      str(beanplot_axis) + ', cex.ylab=' + str(beanplot_yaxis) + ', cex.xlab=' + str(beanplot_xaxis) + ')')

            ro.r("par(las=1)")
            ro.r("dev.off()")


def replace_strange_chars(marker):
    """
    Replace problematic chars for filenames
    :param marker: protein/biomarker name
    :return: sanitized marker name
    """
    strange_chars = ["/", "%", " ", ".", ">", "<", "|", ":", "&", "(", ")", ";", "?", "*"]
    for char in strange_chars:
        marker = marker.replace(char, "_")
    return marker


def print_heatmap(proteins, data, labels, unique_labels, samples, scaling, filename, heatmap_width, heatmap_height,
                  heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar):
    """
    :param proteins: proteins (rows)
    :param data: data as table
    :param labels: labels
    :param unique_labels: tags
    :param samples: samples (columns)
    :param filename: name of file to be created
    :param scaling: column or row scaling
    :param heatmap_width: width of the heatmap png
    :param heatmap_height: heigth of the heatmap png
    :param heatmap_clust_features: list: 0 = hierarchical or not, 1 = metric ("euclidean","manhattan","maximum"),
    2=linkage_method
    :param heatmap_clust_samples: list: 0 = hierarchical or not, 1 = metric, 2=linkage_method ("average","single",
    "complete","centroid","ward.D")
    :param heatmap_zscore_bar: boolean,  bar of zscore's amplitude to be plotted or not
    :return: None, creates a heatmap
    """

    samples_colors = list()
    col_palet = ["#FF2D2D", "#78C87A", "#FFD966", "#3388CC", "#00FFFF", "#00000F", "#F0000F", "#FF000F", "#FFF00F",
                 "#FFFF0F"]
    # col_palet = ["#FF2D2D", "#0000FF", "#FFFFFF"]
    col_palet = ["purple", "orange", "white"]
    if len(unique_labels) <= 10:
        max_num_of_colors = len(unique_labels)
    else:
        max_num_of_colors = 10
    for i in range(len(data[0])):
        for k in range(max_num_of_colors):
            if labels[i] == unique_labels[k]:
                samples_colors.append(col_palet[k])

    # features are proteins and samples are labels
    logging.debug('Heatmaps as PNG')
    r.png(filename, width=heatmap_width, height=heatmap_height, units='cm', res=600)
    # change r.png(filename, width = 10, height = 10,units = 'in',res = 1200)
    data_array = np.asarray(data)
    r("library(gplots)")

    ro.r.assign('data', ro.r.matrix(data_array, nrow=len(data)))
    ro.r.assign('samples', ro.r.matrix(np.asarray(samples), nrow=len(samples)))
    ro.r.assign('proteins', ro.r.matrix(np.asarray(proteins), nrow=len(proteins)))
    ro.r.assign('samples_colors', ro.r.matrix(np.asarray(samples_colors), nrow=len(samples_colors)))

    # depending on the number of the features and samples, different sizes are given to labels
    if len(proteins) <= 100:
        ro.r.assign("row_size", 0.8)
    elif len(proteins) <= 200:
        ro.r.assign("row_size", 1)
    elif len(proteins) <= 300:
        ro.r.assign("row_size", 0.8)
    elif len(proteins) <= 600:
        ro.r.assign("row_size", 0.5)
    elif len(proteins) <= 1000:
        ro.r.assign("row_size", 0.3)
    else:
        ro.r.assign("row_size", 0.2)

    if len(samples) <= 20:
        ro.r.assign("col_size", 1.5)
    elif len(samples) <= 30:
        ro.r.assign("col_size", 1)
    elif len(samples) <= 50:
        ro.r.assign("col_size", 0.5)
    else:
        ro.r.assign("col_size", 0.3)

    if heatmap_clust_features[0] == "hierarchical" and heatmap_clust_samples[0] == "hierarchical":
        logging.debug("Hierarchical clustering of features and samples")
        ro.r.assign("metric_features", heatmap_clust_features[1])  # "euclidean","manhattan","maximum"
        ro.r.assign("linkage_method_features",
                    heatmap_clust_features[2])  # "average","single","complete","centroid","ward.D"

        ro.r.assign("metric_samples", heatmap_clust_samples[1])  # "euclidean","manhattan","maximum"
        ro.r.assign("linkage_method_samples",
                    heatmap_clust_samples[2])  # "average","single","complete","centroid","ward.D"

        ro.r("""
			#use given metric and linkage method for hierarchical clustering 
			features_distance_matrix = dist(data, method=metric_features)
			samples_distance_matrix = dist(t(data), method=metric_samples)
			features_dend = as.dendrogram(hclust(features_distance_matrix, method=linkage_method_features))
			samples_dend = as.dendrogram(hclust(samples_distance_matrix, method=linkage_method_samples))

			#delete every second feature name so they are clear in the heatmap
			if (length(proteins)> 200){
				proteins <- proteins[seq(2,length(proteins),2)]
			}
		""")
        if heatmap_zscore_bar == 1:
            ro.r(
                'heatmap.2(data, Rowv = features_dend, Colv = samples_dend, cexCol = col_size,cexRow=row_size,ColSideColors = samples_colors,labCol=FALSE,'
                'labRow=proteins, col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none")')
        else:
            ro.r(
                'heatmap.2(data, Rowv = features_dend, Colv = samples_dend, cexCol = col_size,cexRow=row_size,labCol=FALSE,'
                'labRow=proteins, col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none",key=FALSE)')

    elif heatmap_clust_features[0] == "hierarchical" and not heatmap_clust_samples[0] == "hierarchical":
        logging.debug("Hierarchical clustering of features but not samples")

        ro.r.assign("metric_features", heatmap_clust_features[1])  # "euclidean","manhattan","maximum"
        ro.r.assign("linkage_method_features",
                    heatmap_clust_features[2])  # "average","single","complete","centroid","ward.D"
        ro.r("""
			features_distance_matrix = dist(data, method=metric_features)
			features_dend = as.dendrogram(hclust(features_distance_matrix, method=linkage_method_features))

			if (length(proteins)> 200){
				proteins <- proteins[seq(2,length(proteins),2)]
			}
		""")

        if heatmap_zscore_bar == 1:
            ro.r(
                'heatmap.2(data, Rowv = features_dend, Colv = FALSE,dendrogram="row", cexCol = col_size,cexRow=row_size, labCol= FALSE,labRow=proteins,ColSideColors = samples_colors,'
                ' col=colorRampPalette(c("blue", "white", "red"))(n=75) ,trace="none")')
        else:
            ro.r(
                'heatmap.2(data, Rowv = features_dend, Colv = FALSE,dendrogram="row", cexCol = col_size,cexRow=row_size, labCol=FALSE,labRow=proteins,'
                ' col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none",key=FALSE)')

    elif not heatmap_clust_features[0] == "hierarchical" and heatmap_clust_samples[0] == "hierarchical":
        ro.r.assign("metric_samples", heatmap_clust_samples[1])  # "euclidean","manhattan","maximum"
        ro.r.assign("linkage_method_samples",
					heatmap_clust_samples[2])  # "average","single","complete","centroid","ward.D"

        ro.r("""
			samples_distance_matrix = dist(t(data), method=metric_samples)
			samples_dend = as.dendrogram(hclust(samples_distance_matrix, method=linkage_method_samples))

			if (length(proteins)> 200){
				proteins <- proteins[seq(2,length(proteins),2)]
			}
		""")

        if heatmap_zscore_bar == 1:
            ro.r(
				'heatmap.2(data, Rowv = FALSE, Colv = samples_dend,dendrogram="column", cexCol = col_size,cexRow=row_size, labCol=FALSE,labRow=proteins,ColSideColors = samples_colors,'
				'col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none")')
        else:
            ro.r(
				'heatmap.2(data, Rowv = FALSE, Colv = samples_dend,dendrogram="column", cexCol = col_size,cexRow=row_size, labCol=FALSE,labRow=proteins,'
                'col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none",key=FALSE)')

    elif heatmap_clust_features[0] == "none" and heatmap_clust_samples[0] == "none":
        ro.r("""
			if (length(proteins)> 200){
				proteins <- proteins[seq(2,length(proteins),2)]
			}
			""")

        if heatmap_zscore_bar == 1:
            ro.r(
                'heatmap.2(data,Rowv=FALSE,Colv=FALSE,dendrogram="none", cexCol = col_size,cexRow=row_size, labCol=FALSE,labRow=proteins,ColSideColors = samples_colors,'
                ' col=colorRampPalette(c("red", "blue", "white"))(n=75),trace="none")')
        else:
            ro.r(
                'heatmap.2(data,Rowv=FALSE,Colv=FALSE,dendrogram="none", cexCol = col_size,cexRow=row_size, labCol=FALSE,labRow=proteins,'
                ' col=colorRampPalette(c("red", "blue", "white"))(n=75),trace="none",key=FALSE)')

    # r.heatmap.2(data_array, cexRow=scaling, labCol=np.asarray(samples),labRow=np.asarray(proteins),ColSideColors =
    # np.asarray(samples_colors), col = ro.r.redgreen(75), show_heatmap_legend = True, show_annotation_legend = True)
    # r.heatmap(data_array, labCol=np.asarray(samples),labRow=np.asarray(proteins),ColSideColors =
    # np.asarray(samples_colors), col=ro.r.colorRampPalette(ro.r.c("red", "yellow", "green"))(n=75))
    # r('heatmap(data, cexRow=scaling, labCol=samples, labRow=proteins,ColSideColors=samples_colors,
    # col=colorRampPalette(c("red", "yellow", "green"))(n=75))')
    # ro.r('legend("topright", legend = (unique_labels), col = (as.numeric(unique_labels)), lty= 1, lwd = 5, cex=.7)')
    r("dev.off()")


def wilcoxon_rank_sum_test(markers, control, condition, paired_flag):
    """
    Calculate  Wilcoxon rank-sum test for paired data, and signed-rank test for unpaired
    :param markers: list of features/markers
    :param control: 2d list with values of control samples
    :param condition: 2d list with values of 2nd condition samples
    :param paired_flag: if we want paired on upaired analysis
    :return: list with p-values, list with fold change and list with standard deviation
    """
    pvals = list()
    Zs = list()
    upregulated_proteins = list()
    downregulated_proteins = list()
    num_of_ups = 0
    num_of_downs = 0
    folds = list()
    stdevs = list()

    # change
    average_control = list()
    average_condition = list()

    for i in range(len(markers)):

        control_data_per_protein = list()
        condition_data_per_protein = list()
        for j in range(len(control[i])):
            control_data_per_protein.append(control[i][j])
        for j in range(len(condition[i])):
            condition_data_per_protein.append(condition[i][j])

        # change
        # creating the 2 lists of averages of control and condition data
        average_control.append(statistics.mean(control_data_per_protein))  # CI.L
        average_condition.append(statistics.mean(condition_data_per_protein))  # CI.R

        # [z,pval]=st.wilcoxon(control_data_per_protein, condition_data_per_protein)
        try:
            if statistics.stdev(control_data_per_protein) == 0 and statistics.stdev(condition_data_per_protein) == 0 \
                    and statistics.mean(control_data_per_protein) == statistics.mean(condition_data_per_protein):
                pval = 1
            else:
                if paired_flag == 0:
                    [z, pval] = st.ranksums(control_data_per_protein, condition_data_per_protein)
                else:
                    [z, pval] = st.wilcoxon(control_data_per_protein, condition_data_per_protein)
            pvals.append(pval)
        except statistics.StatisticsError:
            pval = 1

        if paired_flag == 1:
            fold_per_sample = list()

            for k in range(len(condition_data_per_protein)):
                fold_per_sample.append(condition_data_per_protein[k] - control_data_per_protein[k])
        try:
            if statistics.mean(control_data_per_protein) == 0:
                folds.append(0)
                stdevs.append(0)
            else:
                folds.append(statistics.mean(condition_data_per_protein) - statistics.mean(control_data_per_protein))
                if paired_flag == 1:
                    stdevs.append(statistics.stdev(fold_per_sample))
                else:
                    stdevs.append(0)
        except TypeError:
            folds.append(0)
            stdevs.append(0)
    if paired_flag == 0:
        stdevs = [0] * len(markers)

    return [pvals, folds, stdevs]


def t_test(markers, control, condition, paired_flag):
    """
    Calculate the t-test on TWO RELATED samples of scores, a and b. for paired data, and Calculate the T-test for
    the means of two independent samples of scores. for unpaired data
    :param markers: list of features/markers
    :param control: 2d list with values of control samples
    :param condition: 2d list with values of 2nd condition samples
    :param paired_flag: if we want paired on upaired analysis
    :return: list with p-values, list with fold change and list with standard deviation
    """
    pvals = list()
    Zs = list()
    upregulated_proteins = list()
    downregulated_proteins = list()
    num_of_ups = 0
    num_of_downs = 0
    folds = list()
    stdevs = list()
    for i in range(len(markers)):

        control_data_per_protein = list()
        condition_data_per_protein = list()
        for j in range(len(control[i])):
            control_data_per_protein.append(control[i][j])
        for j in range(len(condition[i])):
            condition_data_per_protein.append(condition[i][j])
        try:
            if statistics.stdev(control_data_per_protein) == 0 and statistics.stdev(condition_data_per_protein) == 0 \
                    and statistics.mean(control_data_per_protein) == statistics.mean(condition_data_per_protein):
                pval = 1
            else:
                if paired_flag == 1:
                    [z, pval] = st.ttest_rel(control_data_per_protein, condition_data_per_protein)
                else:
                    [z, pval] = st.ttest_ind(control_data_per_protein, condition_data_per_protein)
        except statistics.StatisticsError:
            pval = 1
        except TypeError:
            pval = 1

        pvals.append(pval)
        if paired_flag == 1:
            fold_per_sample = list()

            for k in range(len(condition_data_per_protein)):
                fold_per_sample.append(condition_data_per_protein[k] - control_data_per_protein[k])
        try:
            if statistics.mean(control_data_per_protein) == 0:
                folds.append(0)
                stdevs.append(0)
            else:
                folds.append(statistics.mean(condition_data_per_protein) - statistics.mean(control_data_per_protein))
                if paired_flag == 1:
                    stdevs.append(statistics.stdev(fold_per_sample))
                else:
                    stdevs.append(0)
        except TypeError:
            folds.append(0)
            stdevs.append(0)
    if paired_flag == 0:
        stdevs = [0] * len(markers)
    return [pvals, folds, stdevs]


def run_statistical_analysis(data, labels, markers, samples, commorbidities_filename, commorbidities_types_filename,
                             parametric_flag, p_value_threshold, paired_flag, logged_flag, folder_name,
                             volcano_width, volcano_height, volcano_titles, volcano_axis_labels, volcano_labels,
                             volcano_axis_relevance, volcano_criteria, abs_log_fold_changes_threshold, volcano_labeled,
                             heatmap_width, heatmap_height, heatmap_clust_features, heatmap_clust_samples,
                             heatmap_zscore_bar, beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis,
                             beanplot_yaxis, beanplot_titles, beanplot_axis_titles, pid, jobid, user, image_flag=True):
    """
    Run Statistc Analysis for Two condition/class dataset, perform the selected parametric or nonparametric testing,
    and paired or unpaired analysis, create the various images and calculate the different Molecule Quantification files
    :param data: the input dataset as 2d list, rows of features, columns of samples
    :param labels: list of labels per sample
    :param markers: list of features names
    :param samples: list of sample names
    :param commorbidities_filename:
    :param commorbidities_types_filename: file with commorbidities
    :param parametric_flag: 0 if you don't know if you want parametric or non-parametric testing to be applied,
    1 if you wish parametric testing to be applied, 2 if you wish non-parametric testing wih wilcoxon-sum to be
    applied, 3 if you want parametric testing with t-test
    :param p_value_threshold:  p value threshold, used for filtering
    :param paired_flag: 1 for paired analysis 0 for non paired analysis (needed for wilcoxon rank sum)
    :param logged_flag: 0 if data not logged, 1 if logged
    :param folder_name: Older name for output files
    :param volcano_width: A numeric input (size of volcano png width in cm)
    :param volcano_height: A numeric input (size of volcano png height in cm)
    :param volcano_titles: A numeric input  (size of all volcano titles)
    :param volcano_axis_labels: A numeric input  (size of axis titles)
    :param volcano_labels: A numeric input (size of labels names)
    :param volcano_axis_relevance: A numeric input (the relevance between the 2 axis. 32 could be the the default
    because it fills the whole png)
    :param volcano_criteria: A numeric input (between 1,2 and 3)
    :param abs_log_fold_changes_threshold: A numeric (float) number between (0 and 1)
    :param volcano_labeled: A numeric input (between 0,1 and 2)
    :param heatmap_width: A numeric input (size of heatmap png width in cm)
    :param heatmap_height: A numeric input (size of heatmap png height in cm)
    :param heatmap_clust_features: List with 3 fields, 1 An option between "hierarchical" and "none", 2 An option
    between "euclidean","manhattan","maximum" and 3 An option between "average","single","complete","centroid","ward.D"
    :param heatmap_clust_samples: List with 3 fields, 1 An option between "hierarchical" and "none", 2 An option
    between "euclidean","manhattan","maximum" and 3 An option between "average","single","complete","centroid","ward.D"
    :param heatmap_zscore_bar: An option between 1 and 0 (to show the z-score bar or not)
    :param beanplot_width: A numeric input (size of beanplot png width in cm)
    :param beanplot_height: A numeric input (size of beanplot png height in cm)
    :param beanplot_axis: A numeric input ( scaling of axis (scaling relative to default, e.x. 1=default, 1.5 is 50%
    larger, 0.5 is 50% smaller) )
    :param beanplot_xaxis: A numeric input ( scaling of axis x (scaling relative to default, e.x. 1=default, 1.5 is
    50% larger, 0.5 is 50% smaller) )
    :param beanplot_yaxis: A numeric input ( scaling of axis y (scaling relative to default, e.x. 1=default, 1.5 is 50%
    larger, 0.5 is 50% smaller) )
    :param beanplot_titles: A numeric input ( scaling of titles (scaling relative to default, e.x. 1=default, 1.5 is
    50% larger, 0.5 is 50% smaller) )
    :param beanplot_axis_titles: A numeric input ( scaling of axis titles (scaling relative to default, e.x. 1=default,
    1.5 is 50% larger, 0.5 is 50% smaller) )
    :param user: this job's user
    :param jobid: this job's id
    :param pid: this job's PID
    :return:
    """
    try:
        # parsing

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        commorbidities_types = list()
        commorbidities_flag = 0
        with open(commorbidities_types_filename) as commorbidities_types_fname:
            for line in csv.reader(commorbidities_types_fname, delimiter="\t"):
                for i in range(len(line)):
                    commorbidities_types.append(line[i].strip())
        if len(commorbidities_types) > 0:
            commorbidities_flag = 1

        # fixing: The order is now every time the same and the phenotypes are alphabetically ordered
        unique_labels = sorted(list(set(labels)))

        logging.debug(unique_labels)

        if commorbidities_flag != 0:
            [commorbidities, commorbidities_flag] = parse_commorbidities(samples, commorbidities_filename,
                                                                         commorbidities_types)
        else:
            commorbidities = list()
        # print("test1")
        # print(len(commorbidities))

        output_message = ''
        if commorbidities_flag == 0:
            output_message += 'No commorbidities were provided\n'
        else:
            output_message += str(len(commorbidities)) + ' commorbidities were provided.\n'
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during "
                          "parsing.".format(pid, jobid, user))
        return [0, "Statistical analysis raised the exception during parsing: {}".format(str(e))]
    dataset_imputed = data
    try:
        if parametric_flag == '0':
            data_a = list()
            data_b = list()
            for i in range(len(dataset_imputed)):
                for j in range(len(dataset_imputed[0])):
                    if dataset_imputed[i][j] != '' and dataset_imputed[i][j] != 1000:
                        if labels[j] == unique_labels[0]:
                            data_a.append(dataset_imputed[i][j])
                        else:
                            data_b.append(dataset_imputed[i][j])
            [shapiro, shapiro_pvals_a] = st.shapiro(data_a)
            [shapiro, shapiro_pvals_b] = st.shapiro(data_b)
            if shapiro_pvals_a > 0.05 and shapiro_pvals_b > 0.05:
                test_flag = 1
                output_message += 'Data are normally distributed. Parametric testing will be done.\n'
                logging.debug('Data are normally distributed. Parametric testing will be done.')
            else:
                test_flag = 2
                output_message += 'Data are not normally distributed. Non parametric testing will be done.\n'
                logging.debug('Data are not normally distributed. Non parametric testing will be done.')

        elif parametric_flag == '1':
            test_flag = 1
            output_message += 'Parametric testing will be done.\n'
            logging.debug('Parametric testing will be done.')

        elif parametric_flag == '2':
            test_flag = 2
            output_message += 'Non Parametric testing will be done.\n'
            logging.debug('Non Parametric testing will be done.')

        else:
            test_flag = 3
            output_message += 'Parametric T-test will be done.\n'
            logging.debug('Parametric T-test will be done.')

        if test_flag == 1:
            if paired_flag == 1:
                counter1 = 0
                counter2 = 0
                if commorbidities_flag == 0:
                    new_com = list()
                    new_com.append([])
                    for j in range(len(dataset_imputed[0])):
                        # print(j)
                        if labels[j] == unique_labels[0]:
                            new_com[0].append(counter1)
                            counter1 += 1
                        else:
                            new_com[0].append(counter2)
                            counter2 += 1
                    commorbidities_flag = 1
                    commorbidities_types = list()
                    commorbidities_types.append('0')
                    commorbidities = copy.deepcopy(new_com)

                else:
                    num_of_com = len(commorbidities)
                    commorbidities.append([])
                    for j in range(len(dataset_imputed[0])):
                        if labels[j] == unique_labels[0]:
                            commorbidities[num_of_com].append(counter1)
                            counter1 += 1
                        else:
                            commorbidities[num_of_com].append(counter2)
                            counter2 += 1
                    commorbidities_flag = 1
                    commorbidities_types.append('0')

            logging.debug("Differential expression analysis")
            # change categories added
            category1 = list()
            category2 = list()

            for i in range(len(dataset_imputed)):
                category1.append([])
                category2.append([])
                for j in range(len(dataset_imputed[i])):
                    if labels[j] == unique_labels[0]:
                        category1[i].append(dataset_imputed[i][j])
                    else:
                        category2[i].append(dataset_imputed[i][j])

            [diff_table, diff_proteins, diff_columns] = differential_expression_analysis_new(
                category1, category2, markers, dataset_imputed, labels, commorbidities_flag, commorbidities,
                commorbidities_types, unique_labels, folder_name, unique_labels[0] + 'VS' + unique_labels[1])
            ord_labels = sorted(list(set(labels)))
            f = lambda x, index: tuple(i[index] for i in x)
            # corrected volcano plot
            if image_flag:
                print_volcano_plots(diff_proteins, f(diff_table, 4), f(diff_table, 0),
                                    folder_name + unique_labels[0] + 'VS' + unique_labels[
                                        1] + '_volcano_plot_corrected.png',
                                    folder_name + unique_labels[0] + 'VS' + unique_labels[
                                        1] + '_volcano_plot_corrected_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                   abs_log_fold_changes_threshold, volcano_labeled)
                # not corrected volcano plot
                print_volcano_plots(diff_proteins, f(diff_table, 3), f(diff_table, 0),
                                   folder_name + unique_labels[0] + 'VS' + unique_labels[1] + '_volcano_plot.png',
                                    folder_name + unique_labels[0] + 'VS' + unique_labels[
                                        1] + '_volcano_plot_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                    abs_log_fold_changes_threshold, volcano_labeled)

            # print_volcano_plots(diff_proteins, diff_table[6], diff_table[0], folder_name+unique_labels[0]+'VS'
            # +unique_labels[1]+'_volcano_plot_corrected.png',folder_name+unique_labels[0]+'VS'+unique_labels[1]+
            # '_volcano_plot_corrected_unlabeled.png',p_value_threshold,volcano_width,volcano_height,volcano_labeled)
            # print_volcano_plots(diff_proteins, diff_table[5], diff_table[0], folder_name+unique_labels[0]+'VS'
            # +unique_labels[1]+'_volcano_plot.png',folder_name+unique_labels[0]+'VS'+unique_labels[1]+
            # '_volcano_plot_unlabeled.png',p_value_threshold,volcano_width,volcano_height,volcano_labeled)
            if image_flag:
                print_beanplots(markers, diff_proteins, dataset_imputed, unique_labels, labels,
                                 folder_name + 'beanplots_significant/', f(diff_table, 4),
                                 p_value_threshold, f(diff_table, 0), beanplot_width, beanplot_height,
                            beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles, beanplot_axis_titles)
                print_beanplots(markers, diff_proteins, dataset_imputed, unique_labels, labels, folder_name + 'beanplots/',
                               f(diff_table, 3), p_value_threshold,
                                 f(diff_table, 0), beanplot_width, beanplot_height,
                                 beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles, beanplot_axis_titles)

            # change: significant_molecules_dataset.tsv creation
            # dataset_imputed = np.transpose(dataset_imputed)

            res = print_significant_data_better(dataset_imputed, diff_proteins, markers, samples, f(diff_table, 3),
                                                p_value_threshold, folder_name)
            if res[0]:
                logging.exception("Exception during printing of significant molecules file. {}".format(res[1]))

            filtered_data = list()
            position = 0
            diff_proteins_corrected = list()
            spearman_correlation_table_filtered = list()
            pvalues_correlation_table_filtered = list()

            for i in range(len(diff_proteins)):
                prot = diff_proteins[i]

                if diff_table[i][3] < p_value_threshold:  # [i][5] shifted by 2 because columns CI.L and CI.R
                    # don't exist yet
                    diff_proteins_corrected.append(prot)
                    ind = markers.index(prot)
                    filtered_data.append([])
                    for j in range(len(dataset_imputed[ind])):
                        filtered_data[position].append(dataset_imputed[ind][j])

                    position += 1
            if position > 1:
                logging.debug("testn")
                if image_flag:
                    print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.5,
                                  folder_name + 'heatmap_significant_not_corrected.png',
                                  heatmap_width, heatmap_height, heatmap_clust_features, heatmap_clust_samples,
                                  heatmap_zscore_bar)
                output_file = open(folder_name + 'all_heatmap_ordered_data.tsv', 'w')
                for k in range(len(markers)):
                    output_file.write(str(markers[k]))
                    output_file.write('\n')
                output_file.close()

            filtered_data = list()
            position = 0
            diff_proteins_corrected = list()
            for i in range(len(diff_proteins)):
                prot = diff_proteins[i]
                if diff_table[i][4] < p_value_threshold:  # [6][i]
                    diff_proteins_corrected.append(prot)
                    ind = markers.index(prot)
                    filtered_data.append([])
                    for j in range(len(dataset_imputed[ind])):
                        filtered_data[position].append(dataset_imputed[ind][j])
                    position += 1
            if position > 1:
                if image_flag:
                    print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.5,
                                  folder_name + 'heatmap_significant_corrected.png',
                                  heatmap_width, heatmap_height, heatmap_clust_features, heatmap_clust_samples,
                                  heatmap_zscore_bar)

        elif test_flag == 2:

            category1 = list()
            category2 = list()
            # print(markers)
            # print(len(markers))
            for i in range(len(dataset_imputed)):
                category1.append([])
                category2.append([])
                for j in range(len(dataset_imputed[i])):
                    if labels[j] == unique_labels[0]:
                        category1[i].append(dataset_imputed[i][j])
                    else:
                        category2[i].append(dataset_imputed[i][j])
            try:
                [pvals, folds, stdevs] = wilcoxon_rank_sum_test(markers, category1, category2, paired_flag)
            except Exception as e:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during "
                                  "wilcoxon rank sum test".format(pid, jobid, user))
                return [0,"Statistical analysis raised the exception during wilcoxon rank sum test: {} Please use "
                          "unpaired analysis.".format(str(e))]

            for k in range(len(pvals)):
                # print(pvals[k])
                if 'nan' in str(pvals[k]):
                    pvals[k] = 1
            pvalues = smm.multipletests(pvals, method='fdr_bh')
            pvals2 = pvalues[1]

            header = 'IDs\tInitial Pvalue\tAdjusted Pvalue\tFold Change\tStandard Deviation of Fold Changes\n'
            all_pvals_data_zipped = zip(markers, pvals, pvals2, folds, stdevs)
            all_pvals_data_zipped = list(all_pvals_data_zipped)
            if not os.path.isdir(folder_name + "top20/"):
                os.mkdir(folder_name + "top20/")

            with open(folder_name + "all_pvals.tsv", "w") as handle, open(
                    folder_name + "top20/" + "all_pvals_top20.tsv", "w") as top20:
                handle.write(header)
                top20.write(header)

                handle.write('\n'.join(map('\t'.join,
                                           [[str(id), str(init_pval), str(adj_pval), str(fold_change), str(stdev)] for
                                            l, (id, init_pval, adj_pval, fold_change, stdev) in
                                            enumerate(all_pvals_data_zipped)])))
                # handle.write("\n".join(map("\t".join, list(map(str,all_pvals_data_zipped)))))
                top20.write('\n'.join(map('\t'.join,
                                          [[str(id), str(init_pval), str(adj_pval), str(fold_change), str(stdev)] for
                                           l, (id, init_pval, adj_pval, fold_change, stdev) in
                                           enumerate(all_pvals_data_zipped) if l < 20])))
            if image_flag:
                print_volcano_plots(markers, pvals2, folds, folder_name + unique_labels[0] + '_VS_' + unique_labels[
                    1] + '_volcano_plot_corrected.png',
                                    folder_name + unique_labels[0] + '_VS_' + unique_labels[
                                        1] + '_volcano_plot_corrected_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                    abs_log_fold_changes_threshold, volcano_labeled)
                # not corrected volcano plot
                print_volcano_plots(markers, pvals, folds,
                                   folder_name + unique_labels[0] + '_VS_' + unique_labels[1] + '_volcano_plot.png',
                                    folder_name + unique_labels[0] + '_VS_' + unique_labels[
                                       1] + '_volcano_plot_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                   volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                  abs_log_fold_changes_threshold, volcano_labeled)

            # print_volcano_plots(markers, pvals2, folds, folder_name+unique_labels[0]+'VS'+unique_labels[1]+
            # '_volcano_plot_corrected.png',folder_name+unique_labels[0]+'VS'+unique_labels[1]+
            # '_volcano_plot_corrected_unlabeled.png',p_value_threshold,volcano_width,volcano_height,volcano_labeled)
            # print_volcano_plots(markers, pvals, folds, folder_name+unique_labels[0]+'VS'+unique_labels[1]+
            # '_volcano_plot.png',folder_name+unique_labels[0]+'VS'+unique_labels[1]+'_volcano_plot_unlabeled.png',
            # p_value_threshold,volcano_width,volcano_height,volcano_labeled)
            if image_flag:
                print_beanplots(markers, markers, dataset_imputed, unique_labels, labels,
                                 folder_name + 'beanplots_significant/', pvals2, p_value_threshold, folds, beanplot_width,
                                 beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles,
                                 beanplot_axis_titles)
                print_beanplots(markers, markers, dataset_imputed, unique_labels, labels, folder_name + 'beanplots/', pvals,
                                 p_value_threshold, folds, beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis,
                                 beanplot_yaxis, beanplot_titles, beanplot_axis_titles)

            # change:
            try:
                print_significant_data_better(dataset_imputed, markers, markers, samples, pvals, p_value_threshold,
                                              folder_name)
            except Exception:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during printing of significant molecules "
                                  "file.".format(pid, jobid, user))

            filtered_data = list()
            position = 0
            diff_proteins_corrected = list()
            for i in range(len(markers)):
                prot = markers[i]
                if pvals[i] < p_value_threshold:
                    diff_proteins_corrected.append(prot)
                    ind = markers.index(prot)
                    filtered_data.append([])
                    for j in range(len(dataset_imputed[ind])):
                        filtered_data[position].append(dataset_imputed[ind][j])
                    position += 1
            if position > 1:
                if position > 1:
                    if image_flag:
                        print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.5,
                                      folder_name + 'heatmap_significant_not_corrected.png', heatmap_width,
                                      heatmap_height, heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar)

            filtered_data = list()
            position = 0
            diff_proteins_corrected = list()
            for i in range(len(markers)):
                prot = markers[i]
                if pvals2[i] < p_value_threshold:
                    diff_proteins_corrected.append(prot)
                    ind = markers.index(prot)
                    filtered_data.append([])
                    for j in range(len(dataset_imputed[ind])):
                        filtered_data[position].append(dataset_imputed[ind][j])
                    position += 1
            if position > 1:
                if position > 1:
                    if image_flag:
                        print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.5,
                                      folder_name + 'heatmap_significant_corrected.png', heatmap_width, heatmap_height,
                                      heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar)

        elif test_flag == 3:  # change else:

            category1 = list()
            category2 = list()

            for i in range(len(dataset_imputed)):
                category1.append([])
                category2.append([])
                for j in range(len(dataset_imputed[i])):
                    if labels[j] == unique_labels[0]:
                        category1[i].append(dataset_imputed[i][j])
                    else:
                        category2[i].append(dataset_imputed[i][j])
            logging.debug("T-test")
            try:
                [pvals, folds, stdevs] = t_test(markers, category1, category2, paired_flag)
            except Exception as e:
                logging.exception(
                    "PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception t-test. Please use unpaired"
                    " analysis.".format(pid, jobid, user))
                return [0,
                        "Statistical analysis raised the exception during t-test: {} Please use unpaired "
                        "analysis.".format(str(e))]

            for k in range(len(pvals)):
                # print(pvals[k])
                if 'nan' in str(pvals[k]):
                    pvals[k] = 1
            pvalues = smm.multipletests(pvals, method='fdr_bh')
            pvals2 = pvalues[1]

            header = 'IDs\tInitial Pvalue\tAdjusted Pvalue\tFold Change\tStandard Deviation of Fold Changes\n'
            all_pvals_data_zipped = zip(markers, pvals, pvals2, folds, stdevs)
            all_pvals_data_zipped = list(all_pvals_data_zipped)
            if not os.path.isdir(folder_name + "top20/"):
                os.mkdir(folder_name + "top20/")

            with open(folder_name + "all_pvals.tsv", "w") as handle, open(
                    folder_name + "top20/" + "all_pvals_top20.tsv", "w") as top20:
                handle.write(header)
                top20.write(header)

                handle.write('\n'.join(map('\t'.join,
                                           [[str(id), str(init_pval), str(adj_pval), str(fold_change), str(stdev)] for
                                            l, (id, init_pval, adj_pval, fold_change, stdev) in
                                            enumerate(all_pvals_data_zipped)])))
                # handle.write("\n".join(map("\t".join, list(map(str,all_pvals_data_zipped)))))
                top20.write('\n'.join(map('\t'.join,
                                          [[str(id), str(init_pval), str(adj_pval), str(fold_change), str(stdev)] for
                                           l, (id, init_pval, adj_pval, fold_change, stdev) in
                                           enumerate(all_pvals_data_zipped) if l < 20])))
            if image_flag:
                print_volcano_plots(markers, pvals2, folds, folder_name + unique_labels[0] + '_VS_' + unique_labels[1] +
                                    '_volcano_plot_corrected.png', folder_name + unique_labels[0] + '_VS_' +
                                    unique_labels[1] + '_volcano_plot_corrected_unlabeled.png',
                             p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                   abs_log_fold_changes_threshold, volcano_labeled)
                # not corrected volcano plot
                print_volcano_plots(markers, pvals, folds,
                                    folder_name + unique_labels[0] + '_VS_' + unique_labels[1] + '_volcano_plot.png',
                                    folder_name + unique_labels[0] + '_VS_' + unique_labels[
                                     1] + '_volcano_plot_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                    abs_log_fold_changes_threshold, volcano_labeled)
                print_beanplots(markers, markers, dataset_imputed, unique_labels, labels,
                                 folder_name + 'beanplots_significant/', pvals2, p_value_threshold, folds, beanplot_width,
                                 beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles,
                                 beanplot_axis_titles)
                print_beanplots(markers, markers, dataset_imputed, unique_labels, labels, folder_name + 'beanplots/', pvals,
                                 p_value_threshold, folds, beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis,
                                 beanplot_yaxis, beanplot_titles, beanplot_axis_titles)
            # change
            try:
                print_significant_data_better(dataset_imputed, markers, markers, samples, pvals, p_value_threshold,
                                              folder_name)
            except Exception:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during printing of significant molecules "
                                  "file.".format(pid, jobid, user))

            filtered_data = list()
            position = 0
            diff_proteins_corrected = list()
            for i in range(len(markers)):
                prot = markers[i]
                if pvals[i] < p_value_threshold:
                    diff_proteins_corrected.append(prot)
                    ind = markers.index(prot)
                    filtered_data.append([])
                    for j in range(len(dataset_imputed[ind])):
                        filtered_data[position].append(dataset_imputed[ind][j])
                    position += 1
            if position > 1:
                if position > 1:
                    if image_flag:
                        print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.5,
                                      folder_name + 'heatmap_significant_not_corrected.png',
                                      heatmap_width, heatmap_height, heatmap_clust_features, heatmap_clust_samples,
                                      heatmap_zscore_bar)

            filtered_data = list()
            position = 0
            diff_proteins_corrected = list()
            for i in range(len(markers)):
                prot = markers[i]
                if pvals2[i] < p_value_threshold:
                    diff_proteins_corrected.append(prot)
                    ind = markers.index(prot)
                    filtered_data.append([])

                    for j in range(len(dataset_imputed[ind])):
                        filtered_data[position].append(dataset_imputed[ind][j])
                    position += 1
            if position > 1:
                if position > 1:
                    if image_flag:
                        print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.5,
                                      folder_name + 'heatmap_significant_corrected.png', heatmap_width, heatmap_height,
                                      heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar)
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during "
                          "testing.".format(pid, jobid, user))
        if str(e) == "Error in .ebayes(fit = fit, proportion = proportion, stdev.coef.lim = stdev.coef.lim,  : \n  " \
                     "No residual degrees of freedom in linear model fits\n":
            return [0, "{}. Please use non parametric method !".format(str(e))]
        else:
            return [0, "Statistical analysis raised the exception: {}".format(str(e))]

    try:
        # Printing the MQ files for the full dataset
        logging.debug("Creating MQ files")
        if not os.path.isdir(folder_name + "MQ_files"):
            os.mkdir(folder_name + "MQ_files")
        create_MQ_files(dataset_imputed, markers, labels, "MQ_", "{0}MQ_files/".format(folder_name))

        # First parsing the significant  molecules dataset
        logging.debug("Creating MQ significant files")
        filename = folder_name + "significant_molecules_dataset.tsv"
        if os.path.isfile(filename):
            if not os.path.isdir(folder_name + "MQ_significant_files"):
                os.mkdir(folder_name + "MQ_significant_files")
            sign_markers, sign_dataset, samples = parse_data(filename)
            # print(sign_markers)
            # print(samples)
            # Printing the MQ files for the significant dataset
            create_MQ_files(sign_dataset, sign_markers, labels, "MQ_significant_",
                            "{0}MQ_significant_files/".format(folder_name))
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised an exception during creation of MQ"
                          " files.".format(pid, jobid, user))
        return [0, "Statistical analysis raised an exception during creation of MQ files: {}. ".format(str(e))]

    try:
        logging.debug("creating diff. exp files")
        if os.path.isfile(filename):
            # Creating the differential expression file
            if not os.path.isdir(folder_name + "diff_expression_files"):
                os.mkdir(folder_name + "diff_expression_files")
            input_files = os.listdir(folder_name + "MQ_significant_files/")
            proteins1, dataset1 = new_parse_data(folder_name + "MQ_significant_files/" + input_files[0], "\t")
            proteins2, dataset2 = new_parse_data(folder_name + "MQ_significant_files/" + input_files[1], "\t")
            create_differential_expression_file(dataset1, dataset2, proteins1, logged_flag, p_value_threshold,
                                                test_flag,
                                                folder_name + "diff_expression_files/",
                                                "diff_express_file_{0}_VS_{1}.tsv".format(input_files[0][:-4],
                                                                                          input_files[1][:-4]))
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised an exception during creation of diff."
                          " expression file.".format(pid, jobid, user))
        return [0,
                "Statistical analysis raised an exception during creation of diff. expression file: {}.".format(str(e))]

    if image_flag:
        beanplot_folder = "{}beanplots_significant/".format(folder_name)
        beanplot_archive = "{}beanplots_significant_{}".format(folder_name, jobid)
        shutil.make_archive(beanplot_archive, 'zip', beanplot_folder)
        beanplot_folder = "{}beanplots/".format(folder_name)
        beanplot_archive = "{}beanplots_{}".format(folder_name, jobid)
        shutil.make_archive(beanplot_archive, 'zip', beanplot_folder)

    out_file = open(folder_name + 'info.txt', 'w')
    out_file.write(output_message)
    out_file.close()

    return [1, "Job completed successfully."]


if __name__ == "__main__":
    # python statistical_analysis_v4_3.py carotid_plaques_guhcl_areab_noncalcified_dataset.txt
    # carotid_plaques_guhcl_areab_noncalcified_symptomatic_labels.txt
    # carotid_plaques_guhcl_areab_noncalcified_commorbidities.txt carotid_plaques_sds_commorbidities_types.txt 1 0 0
    # 0.3 0.05 0 calcification_12_2019/guhcl/area_b_noncalcified/symptomatic/ empty_file.txt
    print('')
# Todos:
# 1) Move out parameters which have to do with visualization: Images dimensions, font sizes, pvalue threshold for
# printing labels in volcano plot, binary value for selecting to plot print all labels in volcano plot (using ggrepel:
# code is currently being commented)
# 2) Add an option 3 for doing parametric t-test (paired or unpaired)
# 3) Test and fix correlations against clinical variables
