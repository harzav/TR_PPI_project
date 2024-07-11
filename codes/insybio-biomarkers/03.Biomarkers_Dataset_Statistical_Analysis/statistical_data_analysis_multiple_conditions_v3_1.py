"""
Unused libraries

from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from itertools import compress
from sklearn.preprocessing import Imputer
"""
import os
import statistics
import numpy as np
import scipy.stats as st
import time
import math
import sys
import csv
from numpy import matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smm
import logging
import itertools
import shutil

from scipy.stats import mstats
import copy
from sklearn.cluster import DBSCAN
from sklearn import metrics

from knnimpute import (
    knn_impute_few_observed,
    knn_impute_with_argpartition,
    knn_impute_optimistic,
    knn_impute_reference,
)

import chart_studio

os.environ['R_HOME'] = "/usr/lib/R/"
os.environ['R_USER'] = "/usr/lib/python3/dist-packages/rpy2/"

from rpy2 import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

ro.numpy2ri.activate()
from rpy2.robjects import r
import rpy2.robjects.lib.ggplot2 as ggplot2

gplots = importr("ggplot2")
limma = importr('limma')
# statmod=importr('statmod')
ggrepel = importr('ggrepel')
lattice = importr('lattice')
beanplot = importr('beanplot')


def create_differential_expression_file(data1: list, data2: list, gene_names: list, logged_flag: int,
                                        sign_threshold: float, method: int, output_folder: str, filename_result: str):
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
    for i in range(0, len(data1) - 1):  # maybe -1 needed
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
            output_file.write(gene_names[p_value_rank.index(i + 1)] + "\t" +
                              str(p_value[p_value_rank.index(i + 1)]) + "\t" + str(avgs1[p_value_rank.index(i + 1)]) +
                              "\t" + str(avgs2[p_value_rank.index(i + 1)]) + "\t" +
                              str(fold_changes[p_value_rank.index(i + 1)]) + "\n")
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
    logging.debug('Data were successfully parsed!')
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


def parse_data_all_headers(data_filename):
    """
    Parses data.

    Args:
        data_filename: dataset filename

    Returns: a list of three lists, [proteins, data, samples].
    """
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
                        if line[j] != '':
                            data[num_of_lines - 1].append(float(line[j]))
                        else:
                            data[num_of_lines - 1].append('')
            num_of_lines += 1
    # print('Data were successfully parsed!')
    logging.debug('Data were successfully parsed inside stat analysis single!')
    return [proteins, data, samples]


def get_pairs_from_list(alist: list) -> list:
    """
    Gets all pairs from a list and returns them in a list of unique pairs.

    Args:
        alist (list): the input list

    Returns:
        all_pairs(list): the list of unique pairs
    """
    all_pairs = list()
    for pair in itertools.combinations(alist, 2):
        all_pairs.append(pair)
    return all_pairs


def print_data(data, markers, labels, folder_name, filename):
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
    file = open(folder_name + filename, 'w')
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
    file.close()


def print_significant_data(data, markers1, markers2, samples, pvals, p_value_threshold, output_folder):
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
        raise ValueError("The list new_data is empty, problem with pvalue threshold.")
    else:
        print_data(new_data, new_proteins, samples, output_folder, "significant_molecules_dataset.tsv")


def print_beanplots(new_proteins, proteins, data, values_list, labels_1, folder, pvals, p_value_threshold,
                    fold_changes, beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis,
                    beanplot_titles, beanplot_axis_titles):
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
            splitted_table = list()
            for i, label in enumerate(values_list):
                temp_table = list()

                for k in range(len(labels_1)):

                    if labels_1[k] == label:

                        ind = new_proteins.index(proteins[j])
                        if data[k][ind] != '' and data[k][ind] != -1000 and not np.isnan(data[k][ind]):
                            num_of_cat[0] += 1
                            temp_table.append(float(data[k][ind]))
                splitted_table.append(temp_table)
                ro.r.assign(str(label), ro.r.matrix(np.asarray(temp_table), nrow=len(temp_table)))

            cases = ','.join(values_list)
            protein_name = replace_strange_chars(proteins[j])
            ro.r.png(folder + protein_name + '_beanplot.png', width=beanplot_width, height=beanplot_height, units='cm',
                     res=600)

            # ro.r.assign('control',ro.FloatVector(np.asarray(control_table)))
            # ro.r.assign('case',ro.FloatVector(np.asarray(case_table)))

            ro.r.assign("control", ro.r.matrix(np.asarray(splitted_table), nrow=len(splitted_table)))
            # ro.r.assign("case",ro.r.matrix(np.asarray(case_table),nrow=len(case_table)))

            ro.r('library("beanplot")')

            # ax.set_ylabel('Normalized Quantities')
            # ax.set_xlabel(values_list[0]+' vs '+values_list[1]+'\nPvalue:'+str(round(pvals[j],6)))
            # labels = [item.get_text() for item in ax.get_xticklabels()]
            # labels[0] = values_list[0]+'\n(n=' +str(num_of_cat[0])+')'
            # labels[1] = values_list[1]+'\n(n=' +str(num_of_cat[1])+')'

            new_list = list()
            for i in range(len(values_list)):
                # new_list.append(' \n \n \n\n'+values_list[i]+'\n(n=' +str(num_of_cat[i])+')')
                new_list.append(values_list[i])  # new_list.append('  \n'+values_list[i])
            ro.r.assign('labels_of_comparison', ro.StrVector(values_list))
            # print(ro.FactorVector(values_list))
            # print(ro.StrVector(values_list))

            ro.r("par(las=1,cex.axis=1.8,cex.lab=1.8, font.lab=1.8, mar=c(4,5,4,3)+0.1)")
            try:
                ro.r('beanplot({}, bw="nrd0", log = "", names=labels_of_comparison)'.format(cases))
            # ro.r('beanplot(control, bw="nrd0", log = "")')
            except rpy2.rinterface_lib.embedded.RRuntimeError:
                logging.info(splitted_table)

            # ro.r(
            #     'title("' + str(proteins[j]) + '",xlab="Adj. P-value: '
            #     + str(round(pvals[j], 4)) + '", ylab="Log2(RQ)", cex.main=' + str(beanplot_titles) + ', cex.axis='
            #     + str(beanplot_axis_titles) + ', cex.labs=' + str(beanplot_axis) + ', cex.ylab=' + str(beanplot_yaxis)
            #     + ', cex.xlab=' + str(beanplot_xaxis) + ')')
            ro.r(
                'title("' + str(proteins[j]) + '",xlab="Q-value < 0.0001", ylab="Log2(RQ)", cex.main=' + str(beanplot_titles) + ', cex.axis='
                + str(beanplot_axis_titles) + ', cex.labs=' + str(beanplot_axis) + ', cex.ylab=' + str(beanplot_yaxis)
                + ', cex.xlab=' + str(beanplot_xaxis) + ')')
            # ro.r('title("'+str(proteins[j])+'",xlab="", ylab="", cex.main=2.0,cex.axis=1.8, cex.labs=1.8,
            # cex.ylab=1.8, cex.xlab=1.8)')
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
    :param volcano_axis_relevance: the relevance of axis x to axis y , 1 -> default, e.x. 2 -> axis x twice as bigger
    as axis y
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

    if max(log_pvals) == min(log_pvals):
        volcano_axis_relevance = ((limit2 - limit1) / (0.1)) * volcano_axis_relevance
    else:
        volcano_axis_relevance = ((limit2 - limit1) / (max(log_pvals) - min(log_pvals))) * volcano_axis_relevance
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

    if volcano_labeled == 0:  # unlabeled

        r.png(filename2, width=volcano_width, height=volcano_height, units='cm', res=600)  # change width=7, height=6

        ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
        ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')
        gp = ggplot2.ggplot(data=df) + ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids',
                                                          colour='colors') + \
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
        # unlabeled plot
        ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
        ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')

        r.png(filename, width=volcano_width, height=volcano_height, units='cm', res=600)  # change width=7, height=6
        # labeled
        gp = ggplot2.ggplot(data=df) + ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids',
                                                          colour='colors') + \
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
        # ggplot2.geom_text(colour='black', check_overlap=overlap, size=volcano_labels) + \
        gp.plot()

    r("while (!is.null(dev.list()))  dev.off()")
    # r("dev.off()")
    logging.debug("Volcano plots were successfully created")


def parametric_limma(data, labels, markers, tags):
    """
    Parametric testing using R limma
    :param data: input dataset 2d list
    :param labels: input labels list
    :param markers: input features list
    :param tags: unique labels list
    :return:
    """

    proteins_vector = ro.StrVector(markers)  # all proteins in an one dimensional list

    data_array = np.asarray(data)

    data_robject = ro.r.matrix(data_array, nrow=len(data_array))

    data_robject.rownames = proteins_vector  # all proteins

    labels_array = np.asarray(labels)  # all labels from file
    labels_robject = ro.r.matrix(labels_array, nrow=len(labels_array))  # transform to R object

    ro.r.assign('data', data_array)
    ro.r.assign('labels', labels_robject)

    # no commorbidities are provided in this code
    ro.r('design<-model.matrix(~factor(labels))')

    ro.r('fit <-lmFit (data, design)')
    ro.r('fit2<-eBayes(fit)')

    # change
    ro.r("top<-topTable(fit2,coef=2, number=length(data))")
    table = np.asarray(ro.r("top"))
    # columns = list(ro.r("colnames(top)"))  # logFC	AveExpr		t	P.Value		adj.P.Val	B
    proteins_ids = list(ro.r("rownames(top)"))  # all the proteins

    # Practically pvals and folds are computed first and then the diffs_from_avgs is created, because limma creates
    # the table straight forward and intervention in the code was hard
    # This isn't cost effective so it might as well be changed in the future

    # change: diffs_from_avgs creation
    diffs_from_avgs = list()
    avgs = list()

    for j in range(len(data)):
        diffs_from_avg = list()
        avg_per_type = [0] * len(tags)
        vals_per_type = [0] * len(tags)
        avg = 0
        for i in range(len(tags)):
            for k in range(len(data[0])):
                if labels[k] == tags[i]:
                    vals_per_type[i] += 1
                    avg_per_type[i] += data[j][k]
                    avg += data[j][k]

        avg = avg / float(len(data))
        avgs.append(avg)

        for i in range(len(tags)):
            try:
                avg_per_type[i] = avg_per_type[i] / float(vals_per_type[i])
            except ZeroDivisionError:
                avg_per_type[i] = 0
            if avg_per_type[i] > avg:
                diffs_from_avg.append('+')
            else:
                diffs_from_avg.append('-')
        diffs_from_avgs.append(diffs_from_avg)

    logging.debug("parametric limma test done")
    return [table, proteins_ids, diffs_from_avgs]


def kruskal_wallis_test(data, labels, markers, tags):
    """
    Non-Parametric testing using Kruskal-Wallis statistical test
    :param data: input dataset 2d list
    :param labels: input labels list
    :param markers: input features list
    :param tags: unique labels list
    :return:
    """

    Hs = list()
    pvals = list()
    diffs_from_avgs = list()
    avgs = list()
    folds = list()  # change

    for j in range(len(data[0])):
        diffs_from_avg = list()
        avg_per_type = [0] * len(tags)
        vals_per_type = [0] * len(tags)
        avg = 0
        splitted_data = dict()
        condition_data = dict()  # change

        for i in range(len(tags)):
            splitted_data[i] = list()
            condition_data[i] = list()  # change
            for k in range(len(data)):
                # print(k)
                # print(labels[k])
                if labels[k] == tags[i]:
                    # print("test")
                    vals_per_type[i] += 1
                    avg_per_type[i] += data[k][j]
                    avg += data[k][j]
                    splitted_data[i].append(data[k][j])
                else:  # change: find condition data in order to find the fold changes
                    condition_data[i].append(data[k][j])
        try:
            H, pval = mstats.kruskalwallis(*splitted_data.values())

        except Exception:
            H = 0
            pval = 1
        Hs.append(H)
        pvals.append(pval)
        avg = avg / float(len(data))
        avgs.append(avg)

        # change: computing the folds as in wilcoxon rank sum test
        if statistics.mean(splitted_data) == 0:
            folds.append(0)
        else:
            folds.append(statistics.mean(condition_data) - statistics.mean(splitted_data))

        for i in range(len(tags)):
            avg_per_type[i] = avg_per_type[i] / float(vals_per_type[i])
            if avg_per_type[i] > avg:
                diffs_from_avg.append('+')
            else:
                diffs_from_avg.append('-')
        diffs_from_avgs.append(diffs_from_avg)
    return [Hs, pvals, diffs_from_avgs, folds]


def anova_test(data, labels, markers, tags):
    """
    Parametric testing using ANOVA test
    :param data: input dataset 2d list
    :param labels: input labels list
    :param markers: input features list
    :param tags: unique labels list
    :return:
    """

    Hs = list()
    pvals = list()
    diffs_from_avgs = list()
    avgs = list()
    folds = list()
    
    for j in range(len(data[0])):
        diffs_from_avg = list()
        avg_per_type = [0] * len(tags)
        vals_per_type = [0] * len(tags)
        avg = 0
        splitted_data = dict()
        condition_data = dict()  # change

        for i in range(len(tags)):
            splitted_data[i] = list()
            condition_data[i] = list()  # change
            for k in range(len(data)):
                if labels[k] == tags[i]:
                    vals_per_type[i] += 1
                    avg_per_type[i] += data[k][j]
                    avg += data[k][j]
                    splitted_data[i].append(data[k][j])
                else:
                    condition_data[i].append(data[k][j])
        try:
            H, pval = mstats.f_oneway(*splitted_data.values())
        except Exception:
            H = 0
            pval = 1
        Hs.append(H)
        pvals.append(pval)
        avg = avg / float(len(data))
        avgs.append(avg)

        # change: computing the folds as in wilcoxon rank sum test
        if statistics.mean(splitted_data) == 0:
            folds.append(0)
        else:
            folds.append(statistics.mean(condition_data) - statistics.mean(splitted_data))

        for i in range(len(tags)):
            avg_per_type[i] = avg_per_type[i] / float(vals_per_type[i])
            if avg_per_type[i] > avg:
                diffs_from_avg.append('+')
            else:
                diffs_from_avg.append('-')
        diffs_from_avgs.append(diffs_from_avg)
    return [Hs, pvals, diffs_from_avgs, folds]


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

    control = list(map(list, zip(*control)))
    condition = list(map(list, zip(*condition)))

    for i in range(len(markers)):
        control_data_per_protein = list()
        condition_data_per_protein = list()
        for j in range(len(control[i])):
            control_data_per_protein.append(control[i][j])
        for j in range(len(condition[i])):
            condition_data_per_protein.append(condition[i][j])
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
        except statistics.StatisticsError:
            pval = 1
        pvals.append(pval)
        fold_per_sample = list()

        # change: we choose the least of length between condition_data_per_protein and control_data_per_protein
        # so we don't go out of bounds
        limit = len(condition_data_per_protein) if len(condition_data_per_protein) < len(
            control_data_per_protein) else len(control_data_per_protein)

        for k in range(limit):
            fold_per_sample.append(condition_data_per_protein[k] - control_data_per_protein[k])

        # fold_per_sample = [condition_data_per_protein[k] - control_data_per_protein[k] for k in
        # range(len(condition_data_per_protein))]
        # for k in range(len(condition_data_per_protein)):
        # fold_per_sample.append(condition_data_per_protein[k]-control_data_per_protein[k])
        if statistics.mean(control_data_per_protein) == 0:
            folds.append(0)
            stdevs.append(0)
        else:
            folds.append(statistics.mean(condition_data_per_protein) - statistics.mean(control_data_per_protein))
            if paired_flag == 1:
                stdevs.append(statistics.stdev(fold_per_sample))
            else:
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
    control = list(map(list, zip(*control)))
    condition = list(map(list, zip(*condition)))
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
        pvals.append(pval)
        if paired_flag == 1:
            fold_per_sample = list()

            for k in range(len(condition_data_per_protein)):
                fold_per_sample.append(condition_data_per_protein[k] - control_data_per_protein[k])
        if statistics.mean(control_data_per_protein) == 0:
            folds.append(0)
            stdevs.append(0)
        else:
            folds.append(statistics.mean(condition_data_per_protein) - statistics.mean(control_data_per_protein))
            if paired_flag == 1:
                stdevs.append(statistics.stdev(fold_per_sample))
            else:
                stdevs.append(0)
    if paired_flag == 0:
        stdevs = [0] * len(markers)
    return [pvals, folds, stdevs]


def perform_pairwise_analysis(proteins, data, name_prefix, annotation, tags, parametric_flag, p_value_threshold,
                              output_message, paired_flag, volcano_width, volcano_height, volcano_titles,
                              volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                              abs_log_fold_changes_threshold, volcano_labeled, folder_name):
    """
    Perform Statistic Analysis for each Labels pair of the Multilabel dataset
    :param proteins: list of features/proteins
    :param data: 2d list with dataset (rows features/cols samples)
    :param name_prefix: folder for output files
    :param annotation: list of labels
    :param tags: list of unique labels
    :param parametric_flag: to perform parametric or not analysis, 0/2 for nonparametric and 1/3 for parametric testing
    :param p_value_threshold: p-value threshold for filtering significant data
    :param output_message: string with runtime messages
    :param paired_flag: to perform paired or unpaired analysis
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
    :param folder_name: output folder name
    :return:
    """

    splitted_data = [[[data[k][j] for j in range(len(data[0]))] for k in range(len(data)) if annotation[k] == tags[i]]
                     for i in range(len(tags))]
    """
    splitted_data=list()
    for i in range(len(tags)):
        temp_table=list()
        num_of_samples=0
        for k in range(len(data)):
            if annotation[k]==tags[i]:
                temp_table.append([])
                for j in range(len(data[0])):
                    temp_table[num_of_samples].append(data[k][j])
                num_of_samples+=1
        splitted_data.append(temp_table)
    """

    for i in range(len(tags)):
        for j in range(len(tags)):
            if j > i:
                if parametric_flag == '0' or parametric_flag == '2':
                    output_message += 'Wilcoxon test was used for pairwise comparisons\n'

                    [pvals, folds, stdevs] = wilcoxon_rank_sum_test(proteins, splitted_data[i], splitted_data[j],
                                                                    paired_flag)
                else:
                    [pvals, folds, stdevs] = t_test(proteins, splitted_data[i], splitted_data[j], paired_flag)
                    output_message += 'Paired t-test was used for pairwise comparisons between {} vs {}\n'.format(
                        tags[i], tags[j])
                for k in range(len(pvals)):
                    # print(pvals[k])
                    if 'nan' in str(pvals[k]):
                        pvals[k] = 1
                pvalues = smm.multipletests(pvals, method='fdr_bh')
                pvals2 = pvalues[1]

                print_volcano_plots(proteins, pvals2, folds,
                                    folder_name + tags[i] + '_VS_' + tags[j] + '_volcano_plot_corrected.png',
                                    folder_name + tags[i] + '_VS_' + tags[j] + '_volcano_plot_corrected_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                    abs_log_fold_changes_threshold, volcano_labeled)
                # not corrected volcano plot
                print_volcano_plots(proteins, pvals, folds,
                                    folder_name + tags[i] + '_VS_' + tags[j] + '_volcano_plot.png',
                                    folder_name + tags[i] + '_VS_' + tags[j] + '_volcano_plot_unlabeled.png',
                                    p_value_threshold, volcano_width, volcano_height, volcano_titles,
                                    volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
                                    abs_log_fold_changes_threshold, volcano_labeled)

                header = 'IDs\tInitial Pvalue\tAdjusted Pvalue\tFold Change\tStandard Deviation of Fold Changes\n'
                all_pvals_data_zipped = zip(proteins, pvals, pvals2, folds, stdevs)
                all_pvals_data_zipped = list(all_pvals_data_zipped)
                if not os.path.isdir(name_prefix + "top20/"):
                    os.mkdir(name_prefix + "top20/")

                with open(name_prefix + "all_pvals_" + str(tags[i]) + '_vs_' + str(tags[j]) + '.tsv', "w") as handle,\
                        open(name_prefix + "top20/" + "all_pvals_" + str(tags[i]) + '_vs_' + str(tags[j]) + "top20.tsv",
                             "w") as top20:
                    handle.write(header)
                    top20.write(header)

                    handle.write('\n'.join(map(
                        '\t'.join, [[str(id), str(init_pval), str(adj_pval), str(fold_change), str(stdev)]
                                    for el, (id, init_pval, adj_pval, fold_change, stdev)
                                    in enumerate(all_pvals_data_zipped)])))
                    top20.write('\n'.join(map(
                        '\t'.join, [[str(id), str(init_pval), str(adj_pval), str(fold_change), str(stdev)]
                                    for el, (id, init_pval, adj_pval, fold_change, stdev)
                                    in enumerate(all_pvals_data_zipped) if el < 20])))
    return output_message


def print_heatmap(proteins, data, labels, unique_labels, samples, scaling, filename, heatmap_width, heatmap_height,
                  heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar):
    """
    Print Sample-features heatmap
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
    col_palet = ["green", "purple", "orange", "blue", "white"]

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
    # r.heatmap(data_array)

    ro.r.assign('data', ro.r.matrix(data_array, nrow=len(data)))
    ro.r.assign('samples', ro.r.matrix(np.asarray(samples), nrow=len(samples)))
    ro.r.assign('proteins', ro.r.matrix(np.asarray(proteins), nrow=len(proteins)))
    ro.r.assign('samples_colors', ro.r.matrix(np.asarray(samples_colors), nrow=len(samples_colors)))

    # depending on the number of the features and samples, different sizes are given to labels
    if len(proteins) <= 100:
        ro.r.assign("row_size", 1.5)
    elif len(proteins) <= 200:
        ro.r.assign("row_size", 1)
    elif len(proteins) <= 300:
        ro.r.assign("row_size", 0.8)
    elif len(proteins) <= 600:
        ro.r.assign("row_size", 0.5)
    else:
        ro.r.assign("row_size", 0.1)

    if len(samples) <= 20:
        ro.r.assign("col_size", 1.5)
    elif len(samples) <= 30:
        ro.r.assign("col_size", 1)
    elif len(samples) <= 50:
        ro.r.assign("col_size", 0.5)
    else:
        ro.r.assign("col_size", 0.3)
    ro.r.assign('row_size', 0.75)
    if heatmap_clust_features[0] == "hierarchical" and heatmap_clust_samples[0] == "hierarchical":
        logging.debug("Hierarchical clustering of features and samples")
        ro.r.assign("metric_features", heatmap_clust_features[1])  # "euclidean","manhattan","maximum"
        ro.r.assign("linkage_method_features",
                    heatmap_clust_features[2])  # "average","single","complete","centroid","ward.D"

        ro.r.assign("metric_samples", heatmap_clust_samples[1])  # "euclidean","manhattan","maximum"
        ro.r.assign("linkage_method_samples",
                    heatmap_clust_samples[2])  # "average","single","complete","centroid","ward.D"

        ro.r("""
            # use given metric and linkage method for hierarchical clustering 
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
                'heatmap.2(data, Rowv = features_dend, Colv = samples_dend, cexCol = col_size,cexRow=row_size,ColSideColors = samples_colors,labCol=samples,'
                'labRow=proteins, col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none")')
        else:
            ro.r(
                'heatmap.2(data, Rowv = features_dend, Colv = samples_dend, cexCol = col_size,cexRow=row_size,labCol=samples,'
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
                'heatmap.2(data, Rowv = features_dend, Colv = FALSE, cexCol = col_size,cexRow=row_size, labCol=samples,labRow=proteins,ColSideColors = samples_colors,'
                ' col=colorRampPalette(c("blue", "white", "red"))(n=75) ,trace="none")')
        else:
            ro.r(
                'heatmap.2(data, Rowv = features_dend, Colv = FALSE, cexCol = col_size,cexRow=row_size, labCol=samples,labRow=proteins,'
                ' col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none",key=FALSE)')

    elif not heatmap_clust_features[0] == "hierarchical" and heatmap_clust_samples[0] == "hierarchical":
        logging.debug("Hierarchical clustering of samples but not features")
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
                'heatmap.2(data, Rowv = FALSE, Colv = samples_dend, cexCol = col_size,cexRow=row_size, labCol=samples,labRow=proteins,ColSideColors = samples_colors,'
                'col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none")')
        else:
            ro.r(
                'heatmap.2(data, Rowv = FALSE, Colv = samples_dend, cexCol = col_size,cexRow=row_size, labCol=samples,labRow=proteins,'
                'col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none",key=FALSE)')

    elif heatmap_clust_features[0] == "none" and heatmap_clust_samples[0] == "none":
        logging.debug("Non Hierarchical clustering")
        ro.r("""
            if (length(proteins)> 200){
                proteins <- proteins[seq(2,length(proteins),2)]
            }
            """)

        if heatmap_zscore_bar == 1:
            ro.r(
                'heatmap.2(data,Rowv=FALSE,Colv=FALSE, cexCol = col_size,cexRow=row_size, labCol=samples,labRow=proteins,ColSideColors = samples_colors,'
                ' col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none")')
        else:
            ro.r(
                'heatmap.2(data,Rowv=FALSE,Colv=FALSE, cexCol = col_size,cexRow=row_size, labCol=samples,labRow=proteins,'
                ' col=colorRampPalette(c("blue", "white", "red"))(n=75),trace="none",key=FALSE)')

    # r.heatmap.2(data_array, cexRow=scaling, labCol=np.asarray(samples),labRow=np.asarray(proteins),
    # ColSideColors = np.asarray(samples_colors), col = ro.r.redgreen(75), show_heatmap_legend = True,
    # show_annotation_legend = True)
    # r.heatmap(data_array, labCol=np.asarray(samples),labRow=np.asarray(proteins),ColSideColors =
    # np.asarray(samples_colors), col=ro.r.colorRampPalette(ro.r.c("red", "yellow", "green"))(n=75))
    # r('heatmap(data, cexRow=scaling, labCol=samples, labRow=proteins,ColSideColors=samples_colors,
    # col=colorRampPalette(c("red", "yellow", "green"))(n=75))')

    r("dev.off()")


def run_statistical_data_analysis_multiple_conditions(
        data, labels, markers, samples, commorbidities_filename, commorbidities_types_filename, parametric_flag,
        p_value_threshold, paired_flag, logged_flag, folder_name, volcano_width, volcano_height, volcano_titles,
        volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria, abs_log_fold_changes_threshold,
        volcano_labeled, heatmap_width, heatmap_height, heatmap_clust_features, heatmap_clust_samples,
        heatmap_zscore_bar, beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis,
        beanplot_titles, beanplot_axis_titles, pid, jobid, user, image_flag=True):
    """
    Runs the statistical data analysis script for multiple conditions, perform the selected parametric or nonparametric
    testing, and paired or unpaired analysis, create the various images and calculate the different Molecule
    Quantification files
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
    :return: [1, string1] for executing successfully and [0, string2] for failing to execute.
    """

    output_message = ''
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # data has number_of_samples * timepoints rows and number of markrs columns
    # global tags
    tags = sorted(list(set(labels)))
    data_transp = data

    data = list(map(list, zip(*data_transp)))

    # Step 6 Print Box Plots

    # chart_studio.tools.set_credentials_file(username='theofilk', api_key='FvCv1RgpeQC6ng3rxNfR')

    # change: computing the control and condition data as in two conditions code
    # the dimentions are transpose
    if parametric_flag == '0':
        data_per_phenotype = list()

        shapiro_pvals = list()
        flag = 0
        for k in range(len(tags)):
            data_per_phenotype.append([])
            for i in range(len(data_transp)):
                for j in range(len(data_transp[i])):
                    if labels[j] == tags[k]:
                        data_per_phenotype[k].append(data_transp[i][j])

            [shapiro, shapiro_pvals_temp] = st.shapiro(data_per_phenotype[k])
            shapiro_pvals.append(shapiro_pvals_temp)
            if shapiro_pvals_temp < 0.05:
                flag = 1
        if flag == 1:
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tkruskal_wallis_test".format(pid, jobid, user))
            [Hs, pvals, diffs_from_avgs, folds] = kruskal_wallis_test(data, labels, markers, tags)
        else:
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tanova_test".format(pid, jobid, user))
            [Hs, pvals, diffs_from_avgs, folds] = anova_test(data, labels, markers, tags)
        if flag == 1:
            output_file = open(folder_name + 'kruskal_wallis_test.tsv', 'w')
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tkruskal test".format(pid, jobid, user))
        else:
            output_file = open(folder_name + 'anova_test.tsv', 'w')
            logging.info("PID:{}\tJOB:{}\tUSER:{}\tanova test".format(pid, jobid, user))
        output_file.write('ID\tPvalue\tAdjusted Pvalues\tDifferentiation from the average per samples type\n')
        pvalues = smm.multipletests(pvals, method='fdr_bh')
        pvals2 = pvalues[1]

        for i in range(len(pvals)):
            added_message = ''
            for j in range(len(diffs_from_avgs[i])):
                added_message += '\t' + str(diffs_from_avgs[i][j])
            output_file.write(markers[i] + '\t' + str(pvals[i]) + '\t' + str(pvals2[i]) + added_message + '\n')
        output_file.close()

        if flag == 1:
            output_message += 'Kruskal Wallis test was used for multple phenotypes comparison because at least for' \
                              ' one of the categories data are not normally distributed\n'
            parametric_flag = 2
        else:
            output_message += 'Anova test was used for multple phenotypes comparison because for all of' \
                              ' the categories data are normally distributed\n'
            parametric_flag = 1

    elif parametric_flag == '1':
        # change: parametric limma test added
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tparametric test with limma".format(pid, jobid, user))
        [diff_table, diff_proteins, diffs_from_avgs] = parametric_limma(data_transp, labels, markers, tags)

        f = lambda x, index: tuple(i[index] for i in x)
        markers = diff_proteins
        folds = f(diff_table, 0)
        pvals = f(diff_table, 3)
        pvals2 = f(diff_table, 4)

        output_file = open(folder_name + 'parametric_limma_test.tsv', 'w')
        output_file.write('ID\tPvalue\tAdjusted Pvalues\tDifferentiation from the average per samples type\n')
        # pvalues = smm.multipletests(pvals, method='fdr_bh')
        for i in range(len(pvals)):
            added_message = ''
            for j in range(len(diffs_from_avgs[i])):
                added_message += '\t' + str(diffs_from_avgs[i][j])
            output_file.write(markers[i] + '\t' + str(pvals[i]) + '\t' + str(pvals2[i]) + added_message + '\n')
        output_file.close()
        output_message += 'Parametric test with limma\n'

    elif parametric_flag == "2":
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tkruskal test".format(pid, jobid, user))
        [Hs, pvals, diffs_from_avgs, folds] = kruskal_wallis_test(data, labels, markers, tags)
        output_file = open(folder_name + 'kruskal_wallis_test.tsv', 'w')
        output_file.write('ID\tPvalue\tAdjusted Pvalues\tDifferentiation from the average per samples type\n')
        pvalues = smm.multipletests(pvals, method='fdr_bh')
        pvals2 = pvalues[1]
        for i in range(len(pvals)):
            added_message = ''
            for j in range(len(diffs_from_avgs[i])):
                added_message += '\t' + str(diffs_from_avgs[i][j])
            output_file.write(markers[i] + '\t' + str(pvals[i]) + '\t' + str(pvals2[i]) + added_message + '\n')
        output_file.close()
        output_message += 'Kruskal Wallis test was used for multple phenotypes comparison\n'

    else:
        logging.debug('parametric testing without limma')
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tanova test".format(pid, jobid, user))
        [Hs, pvals, diffs_from_avgs, folds] = anova_test(data, labels, markers, tags)

        output_file = open(folder_name + 'anova_test.tsv', 'w')
        output_file.write('ID\tPvalue\tAdjusted Pvalues\tDifferentiation from the average per samples type\n')
        pvalues = smm.multipletests(pvals, method='fdr_bh')
        pvals2 = pvalues[1]
        for i in range(len(pvals)):
            added_message = ''
            for j in range(len(diffs_from_avgs[i])):
                added_message += '\t' + str(diffs_from_avgs[i][j])
            output_file.write(markers[i] + '\t' + str(pvals[i]) + '\t' + str(pvals2[i]) + added_message + '\n')
        output_file.close()
        output_message += 'Anova test was used for multiple phenotypes comparison\n'
    if image_flag:
        print_beanplots(markers, markers, data, tags, labels, folder_name + 'beanplots/', pvals2, 1, folds,
                        beanplot_width, beanplot_height,
                        beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles, beanplot_axis_titles)
        # Step 8 Keep Significant Data Only

        print_beanplots(markers, markers, data, tags, labels, folder_name + 'beanplots_significant/', pvals,
                        p_value_threshold, folds, beanplot_width, beanplot_height, beanplot_axis, beanplot_xaxis,
                        beanplot_yaxis, beanplot_titles, beanplot_axis_titles)

        beanplot_folder = "{}beanplots_significant/".format(folder_name)
        beanplot_archive = "{}beanplots_significant_{}".format(folder_name, jobid)
        shutil.make_archive(beanplot_archive, 'zip', beanplot_folder)

        beanplot_folder = "{}beanplots/".format(folder_name)
        beanplot_archive = "{}beanplots_{}".format(folder_name, jobid)
        shutil.make_archive(beanplot_archive, 'zip', beanplot_folder)
        # change: columns scaling inserted
        # print_heatmap(markers,data_transp,labels,tags, samples,0.4, folder_name+'heatmap_all.png')

        print_heatmap(markers, data_transp, labels, tags, samples, 0.4, folder_name + 'heatmap_all.png', heatmap_width,
                      heatmap_height, heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar)

    if parametric_flag in {0, 1, 2, 3}:
        protein = markers
        proteins2 = markers
        dataset_imputed = data_transp
        filtered_data = list()
        position = 0
        diff_proteins_corrected = list()
        for i, prot in enumerate(protein):
            # prot = protein[i]
            if pvals[i] < p_value_threshold:
                diff_proteins_corrected.append(prot)
                ind = proteins2.index(prot)
                filtered_data.append([])

                for j in range(len(dataset_imputed[ind])):
                    filtered_data[position].append(dataset_imputed[ind][j])
                position += 1
        if position > 1:  # heatmap must have at least 2 rows and 2 columns
            try:
                print_heatmap(diff_proteins_corrected, filtered_data, labels, tags, samples, 0.4,
                              folder_name + "heatmap_not_corrected.png", heatmap_width, heatmap_height,
                              heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar)
            except Exception:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during printing "
                                  "heatmap_not_corrected.png".format(pid, jobid, user))

        filtered_data = list()
        position = 0
        diff_proteins_corrected = list()
        for i in range(len(protein)):
            prot = protein[i]
            if pvals2[i] < p_value_threshold:
                diff_proteins_corrected.append(prot)
                ind = proteins2.index(prot)
                filtered_data.append([])

                for j in range(len(dataset_imputed[ind])):
                    filtered_data[position].append(dataset_imputed[ind][j])
                position += 1
        if position > 1:
            try:
                print_heatmap(diff_proteins_corrected, filtered_data, labels, tags, samples, 0.4,
                              folder_name + "heatmap_corrected.png", heatmap_width, heatmap_height,
                              heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar)
            except Exception:
                logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during printing heatmap "
                                  "corrected.png".format(pid, jobid, user))

    output_file = open(folder_name + 'all_heatmap_ordered_data.tsv', 'w')
    for k in range(len(markers)):
        output_file.write(str(markers[k]))
        for j in range(len(data_transp[k])):
            output_file.write('\t' + str(data_transp[k][j]))
        output_file.write('\n')
    output_file.close()
    # Step 11 Perform Pairwise Statistical Analysis
    output_message = perform_pairwise_analysis(
        markers, data, folder_name, labels, tags, parametric_flag, p_value_threshold, output_message, paired_flag,
        volcano_width, volcano_height, volcano_titles, volcano_axis_labels, volcano_labels, volcano_axis_relevance,
        volcano_criteria, abs_log_fold_changes_threshold, volcano_labeled, folder_name)

    out_file = open(folder_name + 'info.txt', 'w')
    out_file.write(output_message)
    out_file.close()
    # cluster_proteins(markers,data,labels,tags,folder_name+'lps_plasma_clustering',folder_name)
    # change: significant_molecules_dataset.tsv creation

    dataset_imputed = data_transp
    try:
        print_significant_data(dataset_imputed, markers, markers, samples, pvals, p_value_threshold, folder_name)
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tException during printing of significant molecules "
                          "file.".format(pid, jobid, user))
    data = list(map(list, zip(*data)))
    try:
        # Printing the MQ files for the full dataset
        if not os.path.isdir(folder_name + "MQ_files"):
            os.mkdir(folder_name + "MQ_files")
        create_MQ_files(data, markers, labels, "MQ_", "{0}MQ_files/".format(folder_name))

        # First parsing the significant  molecules dataset
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tCreating MQ significant files".format(pid, jobid, user))
        filename = folder_name + "significant_molecules_dataset.tsv"
        if os.path.isfile(filename):
            if not os.path.isdir(folder_name + "MQ_significant_files"):
                os.mkdir(folder_name + "MQ_significant_files")
            sign_markers, sign_dataset, samples = parse_data_all_headers(filename)
            # Printing the MQ files for the significant dataset
            create_MQ_files(sign_dataset, sign_markers, labels, "MQ_significant_",
                            '{0}MQ_significant_files/'.format(folder_name))
    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised an exception during creation of "
                          "MQ files.".format(pid, jobid, user))
        return [0, "Statistical analysis raised an exception during creation of MQ files: {}. ".format(str(e))]

    try:
        logging.info("PID:{}\tJOB:{}\tUSER:{}\tcreating diff. exp files".format(pid, jobid, user))
        if os.path.isfile(filename):
            # Creating differential expression files, for all pairs of MQ files
            if not os.path.isdir(folder_name + "diff_expression_files"):
                os.mkdir(folder_name + "diff_expression_files")
            input_files = os.listdir(folder_name + "MQ_significant_files/")
            pairs_of_names = get_pairs_from_list(input_files)
            for pair in pairs_of_names:
                proteins1, dataset1 = new_parse_data(folder_name + "MQ_significant_files/" + pair[0], "\t")
                proteins2, dataset2 = new_parse_data(folder_name + "MQ_significant_files/" + pair[1], "\t")
                if parametric_flag in {1, 2}:
                    create_differential_expression_file(dataset1, dataset2, proteins1, logged_flag, p_value_threshold,
                                                        parametric_flag,
                                                        folder_name + "diff_expression_files/",
                                                        "diff_express_file_{0}_VS_{1}.tsv".format(pair[0][:-4],
                                                                                                  pair[1][:-4]))
                else:
                    zero_parametric_flag = 1 + int(parametric_flag)  # 1 for parametric, 2 for non parametric
                    create_differential_expression_file(dataset1, dataset2, proteins1, logged_flag, p_value_threshold,
                                                        zero_parametric_flag,
                                                        folder_name + "diff_expression_files/",
                                                        "diff_express_file_{0}_VS_{1}.tsv".format(pair[0][:-4],
                                                                                                  pair[1][:-4]))

    except Exception as e:
        logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis with multiple conditions raised an "
                          "exception during creation of differential expression files.".format(pid, jobid, user))
        return [0, "Statistical analysis with multiple conditions raised an exception during creation of "
                   "differential expression files: {}".format(str(e))]

    return [1, "Job completed successfully."]


if __name__ == "__main__":
    # python3 statistical_data_analysis_multiple_conditions_v3_1.py funcho_ecm_dataset.txt funcho_labels.txt
    # funcho_samples.txt 1 2 2 0.3 0.05 0 funcho_proteomics/ecm_anova/
    print(0)

# Todos:
# 1) Boxplots with Beanplots similarly to the two conditions code
# 2) Export as parameters to the script all parameters for visualizing the heatmaps, volcano plots and beanplots
# 3) Add an option to do parametric testing with anova_test
# 4) Add method limma to be executed when doing parametric testing (option 1) and do corrections for commorbidities
# similarly as done in two conditions testing.
