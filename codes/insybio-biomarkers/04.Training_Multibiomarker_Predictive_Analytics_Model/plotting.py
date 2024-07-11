"""
Draws a plot of an oneline filename and save the result to a PNG file.

Example run:
    python3 plotting.py data_to_plot.txt
"""
import os
from itertools import cycle

import matplotlib
import pandas as pd
from sklearn.preprocessing import label_binarize

matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import logging
from scipy import interp


def parsing_tsv(oneline_filename):
    """
    Parse file, with lines
    :param oneline_filename:  a file with one line, with tab separated numbers
    :return: a list
    """

    features = list()
    with open(oneline_filename, 'r') as features_file:
        for line in features_file:
            word = line.split("\t")
            for w in word:
                features.append(w.rstrip())
    return features


def draw_plots(filename, outfilename):
    """
    Draw plot of best or average performance per generation
    :param filename:
    :param outfilename:
    :return:
    """
    data = parsing_tsv(filename)
    data = [float(x) for x in data]
    plt.plot(data, label='performance per generation')
    plt.xlabel('generation')
    plt.ylabel('performance')
    plt.legend()
    # plt.show()
    plt.savefig(outfilename)
    plt.clf()


def draw_roc_curves(ytest, predicted, flag, outfilename, labels_file):
    """

    Args:
        ytest: testing labels
        predicted: predicted labels in dataframe (pd.get_dummies) format
        flag: flag for selection, 0 for two class, 1 for multiclass
        outfilename: soutput filename for the plot

    Returns:

    """
    labels = open(labels_file)
    labelss = labels.read()
    taste = labelss.split("\t")
    labels.close()
    classes_real = set(taste)
    classes_real = list(classes_real)

    if flag == 0:
        thresh = {}
        y_true = np.array(ytest)
        y_probas = np.array(predicted)

        classes = np.unique(y_true)
        probas = y_probas

        fig, ax = plt.subplots(1, 1,figsize=None)

        ax.set_title('Receiver Operating Characteristic', fontsize='large')
        fpr_d = {}
        tpr_d = {}

        indices_to_plot = np.in1d(classes, classes)
        for i, to_plot in enumerate(indices_to_plot):
            fpr_d[i], tpr_d[i], _ = roc_curve(y_true, probas[:, i],
                                                    pos_label=classes[i])
            if to_plot:
                roc_auc = auc(fpr_d[i], tpr_d[i])
                color = plt.cm.get_cmap('nipy_spectral')(float(i) / len(classes))
                ax.plot(fpr_d[i], tpr_d[i], lw=2, color=color,
                        label='ROC curve of class {0} (area = {1:0.2f})'
                              ''.format(classes_real[i], roc_auc))

        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack(
                    (1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr,
                label='micro-average ROC curve '
                          '(area = {0:0.2f})'.format(roc_auc),
                color='deeppink', linestyle=':', linewidth=4)

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr_d[x] for x in range(len(classes))]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_d[i], tpr_d[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr,
                    label='macro-average ROC curve '
                          '(area = {0:0.2f})'.format(roc_auc),
                    color='navy', linestyle=':', linewidth=4)

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate or (1 - Specificity)', fontsize='medium')
        ax.set_ylabel('True Positive Rate or (Sensitivity)', fontsize='medium')
        ax.tick_params(labelsize='medium')
        font_size = 12
        for x in classes_real:
            if len(x) > 10:
                font_size = 8
                break
        ax.legend(loc='lower right', prop={"size": font_size})
        plt.savefig(outfilename)
        plt.clf()
    return fpr, tpr, roc_auc
if __name__ == "__main__":
    file = sys.argv[1]
    outfile = sys.argv[2]
    draw_plots(file, outfile)

