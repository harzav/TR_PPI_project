import matplotlib
matplotlib.use('Agg')
import os
import statistics
import numpy as np
import scipy.stats as st
import time
import math
import sys
import csv
from numpy import matrix
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
import statsmodels.stats.multitest as smm
import plotly
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import mstats
import copy
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from itertools import compress
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

os.environ['R_HOME'] = "/usr/lib/R"
os.environ['R_USER'] = '/usr/lib/python3/dist-packages/rpy2/'

from rpy2 import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
ro.numpy2ri.activate()
from rpy2.robjects import r
import rpy2.robjects.lib.ggplot2 as ggplot2

import configparser
import datetime
import logging

gplots = importr("gplots")
limma = importr('limma')
statmod=importr('statmod')
lattice=importr('lattice')
from sklearn.preprocessing import Imputer
from knnimpute import (
	knn_impute_few_observed,
	knn_impute_with_argpartition,
	knn_impute_optimistic,
	knn_impute_reference,
)


def outlier_detection(dataset, folder_name):
	'''
	Detects the outliers.

	Args:
		dataset: input dataset
		folder_name: output folder name

	Returns: a list of lists
	'''
	pca = PCA(n_components=0.9,svd_solver = 'full')
	new_dataset=list()
	num_of_samples=0
	for j in range(len(dataset[0])):
		new_dataset.append([])
		for i in range(len(dataset)):
			new_dataset[num_of_samples].append(float(dataset[i][j]))
		num_of_samples+=1
	dataset_new=pca.fit_transform(new_dataset)
	print(len(dataset_new))
	print(len(dataset_new[0]))
	clf = LocalOutlierFactor(n_neighbors=20)
	y_pred = clf.fit_predict(dataset_new)
	write_one_dimensional_list_to_tab_delimited_file(y_pred,folder_name+'outlier_prediction.txt')
	return dataset_new


def parse_data(data_filename):
	'''
	Parses data.

	Args:
		data_filename: dataset filename

	Returns: a list of three lists, [proteins, data, samples].
	'''
	num_of_lines=0
	proteins=list()
	data=list()
	samples=list()
	with open(data_filename) as data_fname:
		for line in csv.reader(data_fname, delimiter="\t"):
			if num_of_lines==0:
				for j in range(len(line)):
					if j>0:
						samples.append(line[j].strip())
			else:
				proteins.append(line[0])
				data.append([])
				for j in range(len(line)):
					if j>0:
						if line[j]!='':
							data[num_of_lines-1].append(float(line[j]))
						else:
							data[num_of_lines-1].append('')
			num_of_lines+=1
	#print('Data were successfully parsed!')
	logging.info('Data were successfully parsed inside stat analysis single!')
	return [proteins,data,samples]


def parse_labels(labels_filename):
	'''
	Parses labels.

	Args:
		labels_filename: the labels filename

	Returns: a list of two lists, [symptomatic_asymptomatic, diabetes].
	'''
	symptomatic_asymptomatic=list()
	diabetes=list()
	num_of_lines=0
	with open(labels_filename) as labels_fname:
		for line in csv.reader(labels_fname, delimiter="\t"):
			if num_of_lines==0:
				for i in range(len(line)):
					if line[i].strip()=='Symptomatic':
						symptomatic_asymptomatic.append('Symptomatic')
					else:
						symptomatic_asymptomatic.append('Asymptomatic')
			else:
				for i in range(len(line)):
					diabetes.append((line[i].strip()))
			num_of_lines+=1
	#print('Labels were successfully parsed!')
	logging.info('Labels were successfully parsed!')
	return [symptomatic_asymptomatic, diabetes]


def parse_labels_new(labels_filename):
	'''
	Parses labels.

	Args:
		labels_filename: the labels filename

	Returns: a list with the labels.
	'''
	labels=list()
	num_of_lines=0
	with open(labels_filename) as labels_fname:
		for line in csv.reader(labels_fname, delimiter="\t"):
			for i in range(len(line)):
				labels.append(line[i].strip())
	#print('Labels were successfully parsed!')
	logging.info('Labels were successfully parsed inside stat analysis single!')
	return labels


def parse_commorbidities(commorbidities_filename):
	'''
	Parses commorbidities.

	Args:
		commorbidities_filename: the filename with the commorbidities

	Returns: a list (commorbidities) and an integer (commorbidities_flag).
	'''
	age=list()
	sex=list()
	statin=list()
	a_or_b=list()
	commorbidities_flag=0
	num_of_lines=0
	commorbidities=list()
	with open(commorbidities_filename) as commorbidities_fname:
		for line in csv.reader(commorbidities_fname, delimiter="\t"):
			commorbidities_flag=1
			if len(commorbidities)==0:
				for i in range(len(line)):
					commorbidities.append([])
			for i in range(len(line)):
				commorbidities[i].append(line[i].strip())
	logging.info("Commorbidities parsed !")
	return commorbidities, commorbidities_flag


def parse_commorbidities_types(commorbidities_types_filename):
	'''
	Parses commorbidities types.

	Args:
		commorbidities_types_filename: the filename of the commorbidities types

	Returns:
		commorbidities_types (list): the list of the commorbidities types
		commorbidities_flag (int): a flag that shows if parsing went OK
	'''
	commorbidities_types=list()
	commorbidities_flag=0
	with open(commorbidities_types_filename) as commorbidities_types_fname:
		for line in csv.reader(commorbidities_types_fname, delimiter="\t"):
			for i in range(len(line)):
				commorbidities_types.append(line[i])
	if len(commorbidities_types)>0:
		commorbidities_flag=1
	logging.info("Commorbitidies types parsed!")
	return commorbidities_types, commorbidities_flag


def filter_proteomics_dataset(dataset_initial, proteins, percentage, output_message):
	'''
	Filters proteomics dataset.

	Args:
		dataset_initial: the initial dataset (a list of lists)
		proteins: a list of proteins
		percentage: a float
		output_message: the output message

	Returns: a list with new_data (filtered data list), new_proteins (filtered proteins), output message (string).
	'''
	new_data=list()
	selected=0
	new_proteins=list()
	missing_proteins=0
	proteins_missing_values_percentage=0
	for i in range(len(dataset_initial)):
		missing=0
		for j in range (len(dataset_initial[0])):
			if dataset_initial[i][j]=='' or dataset_initial[i][j]==-1000:
				missing+=1
				proteins_missing_values_percentage+=1
		if missing/float(len(dataset_initial[0])) < percentage:
			#print(i)
			new_data.append([])
			for k in range(len(dataset_initial[i])):
				new_data[selected].append(dataset_initial[i][k])
			selected+=1
			new_proteins.append(proteins[i])
		else:

			missing_proteins+=1
	print('Data were successfully filtered!')
	print('Total Number of Proteins='+str(len(dataset_initial)))
	print('Total Number of Proteins with missing values less than predefined threshold='+str(selected))
	print('Percentage of Missing Values in all Proteins='+str(proteins_missing_values_percentage/float(len(dataset_initial)*len(dataset_initial[0]))))
	output_message+='Total Number of Molecules='+str(len(dataset_initial))+'\n'
	output_message+='Total Number of Molecules with missing values less than less than allowed threshold='+str(selected)+'\n'
	output_message+='Percentage of Missing Values in all molecules='+str(proteins_missing_values_percentage/float(len(dataset_initial)*len(dataset_initial[0])))+'\n'
	return [new_data,new_proteins,output_message]


def write_list_to_tab_delimited_file(data, filename):
	'''
	Writes list to tab delimited file.

	Args:
		data: input data
		filename: filename to write the input data

	Returns: doesn't return anything, only writes data to file.
	'''
	file_id=open(filename,'w')
	for i in range(len(data)):
		for j in range(len(data[0])):
			file_id.write(str(data[i][j]))
			if j!=len(data[0])-1:
				file_id.write('\t')
			else:
				file_id.write('\n')
	file_id.close()


def write_one_dimensional_list_to_tab_delimited_file(data, filename):
	'''
	Writes one dimensional list to tab delimited file.

	Args:
		data: input data
		filename: output filename

	Returns: doesn't return anything, only writes data to file.
	'''
	file_id=open(filename,'w')
	for i in range(len(data)):
		file_id.write(str(data[i]))
		file_id.write('\n')
	file_id.close()


def differential_expression_analysis(proteins, data, age, sex, statin, labels, suffix):
	'''
	Calculates the differential expression analysis.

	Args:
		proteins: a list with proteins
		data: a list of lists with the data
		age: list with age variables
		sex: list with sex variables
		statin: list with statin variables
		labels: list with labels
		suffix: string suffix for the final resulting name

	Returns: a list with the modelled table (table of top genes from Linear Model fit), with the protein ids (table rownames)
			 and with the columns (table column names).

	Useful links: http://www.ugrad.stat.ubc.ca/R/library/limma/html/toptable.html
				  http://www.ugrad.stat.ubc.ca/R/library/limma/html/ebayes.html
	'''
	proteins_vector=ro.StrVector(proteins)
	data_array=np.asarray(data)
	data_robject=ro.r.matrix(data_array, nrow=len(data_array))
	data_robject.rownames=proteins_vector
	labels_array=np.asarray(labels)
	labels_robject=ro.r.matrix(labels_array,nrow=len(labels_array))
	commorbidity_robject1=ro.r.matrix(np.asarray(age),nrow=len(labels_array))
	commorbidity_robject2=ro.r.matrix(np.asarray(sex),nrow=len(labels_array))
	commorbidity_robject3=ro.r.matrix(np.asarray(statin),nrow=len(labels_array))
	#print(labels_robject)
	#design=model.matrix(ro.FactorVector(labels_robject))
	#print(design)
	ro.r.assign('data',data_robject)
	ro.r.assign('labels', labels_robject)
	ro.r.assign('age',commorbidity_robject1)
	ro.r.assign('sex',commorbidity_robject2)
	ro.r.assign('statin',commorbidity_robject3)

	ro.r('design<-model.matrix(~factor(labels)+age+factor(sex)+factor(statin))')
	ro.r('fit <-lmFit (data, design)')
	ro.r('fit2<-eBayes(fit)')
	fit2=ro.r('eBayes(fit)')
	table=limma.topTable(fit2, coef=2,  n=len(data))
	columns=list(np.asarray(table.colnames))
	proteins_ids=list(np.asarray(table.rownames))
	table_array=(np.asarray(table))

	output_file=open('diff_exp_results'+suffix+'.txt','w')
	output_file.write('\t')
	for title in columns:
		output_file.write(str(title)+'\t')
	output_file.write('\n')
	for i in range(len(table_array[0])):
		output_file.write(str(proteins_ids[i])+'\t')
		for j in range(len(table_array)):
			output_file.write(str(table_array[j][i])+'\t')
		output_file.write('\n')
	output_file.close()

	return [table, proteins_ids, columns]


def differential_expression_analysis_new(proteins, data, labels, commorbidities_flag, commorbidities, commorbidities_type, folder_name):
	'''
	Calculates the differential expression analysis.

	Args:
		proteins: a list with proteins
		data: a list of lists with the data
		labels: list with labels
		commorbidities_flag: commorbidities flag
		commorbidities: list with commorbidities
		commorbidities_type: type of commorbidities (0 or else)
		folder_name: the output folder name

	Returns: a list with the modelled table (table of top genes from Linear Model fit), with the protein ids (table rownames)
			 and with the columns (table column names).

	Useful links: http://www.ugrad.stat.ubc.ca/R/library/limma/html/toptable.html
				  http://www.ugrad.stat.ubc.ca/R/library/limma/html/ebayes.html
	'''
	proteins_vector=ro.StrVector(proteins)
	data_array=np.asarray(data)
	data_robject=ro.r.matrix(data_array, nrow=len(data_array))
	data_robject.rownames=proteins_vector
	labels_array=np.asarray(labels)
	labels_robject=ro.r.matrix(labels_array,nrow=len(labels_array))

	#print(labels_robject)
	#design=model.matrix(ro.FactorVector(labels_robject))
	#print(design)
	ro.r.assign('data',data_robject)
	ro.r.assign('labels', labels_robject)
	if commorbidities_flag==0:
		ro.r('design<-model.matrix(~factor(labels))')
	else:
		command='design<-model.matrix(~factor(labels)'
		for i in range( len(commorbidities)):
			ro.r.assign('commorbidity_'+str(i),ro.r.matrix(np.asarray(commorbidities[i]),nrow=len(labels_array)))
			if commorbidities_type[i]=='0':
				command+='+factor(commorbidity_'+str(i)+')'
			else:
				command+='+commorbidity_'+str(i)
		command+=')'
		ro.r(command)

	#print(ro.r('dim(data)'))
	#print(ro.r('dim(design)'))
	ro.r('fit <-lmFit (data, design)')

	ro.r('fit2<-eBayes(fit)')
	fit2=ro.r('eBayes(fit)')
	table=limma.topTable(fit2, coef=2,  n=len(data))
	columns=list(np.asarray(table.colnames))
	if 'ID' in columns:
		proteins_ids = list(np.asarray(table[0]))
	else:
		proteins_ids=list(np.asarray(table.rownames))
	table_array=(np.asarray(table))

	#print("writing diff exp results to file START")
	output_file=open(folder_name+'diff_exp_results.txt','w')
	output_file.write('\t')
	for i,title in enumerate(columns):
		output_file.write(str(title))
		if i < len(columns) - 1:
			output_file.write('\t')
	output_file.write('\n')
	for i in range(len(table_array[0])):
		output_file.write(str(proteins_ids[i])+'\t')
		for j in range(len(table_array)):
			output_file.write(str(table_array[j][i]))
			if j < len(table_array) - 1:
				output_file.write('\t')
		output_file.write('\n')
	output_file.close()

	os.mkdir(folder_name + "top20/")
	output_file = open(folder_name + "top20/" + "diff_exp_results_top20.txt", 'w')
	output_file.write('\t')
	for i, title in enumerate(columns):
		output_file.write(str(title))
		if i < len(columns) - 1:
			output_file.write('\t')
	output_file.write('\n')
	for i in range(len(table_array[0])):
		if i < 20:
			output_file.write(str(proteins_ids[i]) + '\t')
			for j in range(len(table_array)):
				output_file.write(str(table_array[j][i]))
				if j < len(table_array) - 1:
					output_file.write('\t')
			output_file.write('\n')
	output_file.close()

	logging.info("Differential expression analysis successfully finished!")
	return [table, proteins_ids, columns]


def print_volcano_plots(proteins, pvals, log_fold_changes, filename, p_value_threshold):
	'''
	Prints volcano plots.

	Args:
		proteins: list with proteins
		pvals: list with pvals
		log_fold_changes: log fold changes variable
		filename: output png filename
		p_value_threshold: p value threshold

	Returns: doesn't return anything, only prints volcano plot as a png file.
	'''
	thresholds=list()
	selected_proteins=list()
	for i in range (len(pvals)):
		if pvals[i] <p_value_threshold :
			thresholds.append(1)
			selected_proteins.append(proteins[i])
		else:
			selected_proteins.append('')
			thresholds.append(0)
	ro.FactorVector(thresholds)
	log_pvals=list()
	colors=list()
	for i in range(len(pvals)):
		if pvals[i]<0.001:
			colors.append('red2')
		elif pvals[i]<0.01:
			colors.append('orange1')
		elif pvals[i]<0.05:
			colors.append('darkgreen')
		else:
			colors.append('darkblue')
		log_pvals.append(-1*math.log10(pvals[i]))
	colormap_raw = [['red2', '#ff0000'],['orange', '#FFA500'],['darkgreen', '#006400'],['darkblue', '#003366'] ]
	colormap_labels = [['red2', 'P < 0.001'],['orange', 'P < 0.01'], ['darkgreen', 'P < 0.05'], ['darkblue', 'P > 0.05']]
	colormap = ro.StrVector([elt[1] for elt in colormap_raw])
	colormap.names = ro.StrVector([elt[0] for elt in colormap_raw])
	ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
	ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')
	df_dict = {'Ids':ro.StrVector(selected_proteins),'threshold':ro.IntVector(thresholds),'log2FC': ro.FloatVector(log_fold_changes), 'MinusLog10Pvals': ro.FloatVector(log_pvals),'colors':ro.StrVector(colors)}

	df= ro.DataFrame(df_dict)

	r.png(filename)
	#gp=ggplot2.ggplot(data=df) +ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids', colour='colors') +ggplot2.geom_point() +ggplot2.geom_text(colour ='black', check_overlap='True')+ggplot2.scale_colour_manual("",
	#								values=colormap,
	#								breaks=colormap.names,
	#								labels=[elt[1] for elt in
	#								colormap_labels])
	gp=ggplot2.ggplot(data=df) +ggplot2.aes_string(x='log2FC', y='MinusLog10Pvals', label='Ids', colour='colors') +ggplot2.geom_point() +ggplot2.scale_colour_manual("",
									values=colormap,
									breaks=colormap.names,
									labels=[elt[1] for elt in
									colormap_labels])
	gp.plot()

	ro.r('p1<-expression(paste(-log[10], \"(p-value)\" ))')
	ro.r('p2<-expression(paste(log[2], \"(FC)\" ))')

	r("dev.off()")
	logging.info("Volcano plots successfully printed!")


def print_boxplots(new_proteins, proteins, data, values_list, labels_1, folder, pvals, p_value_threshold):
	'''
	Print boxplots.

	Args:
		new_proteins: new proteins list
		proteins: proteins list
		data: input data, as a list of lists
		values_list: specific values that the labels_1 list can take (used for filtering)
		labels_1: list of labels
		folder: output folder
		pvals: pvals list
		p_value_threshold: p value threshold parameter

	Returns: doesn't return anything, only prints boxplots as png.
	'''
	for j in range(len(proteins)):
		num_of_cat=list()
		num_of_cat.append(0)
		num_of_cat.append(0)
		if pvals[j]<p_value_threshold:
			splitted_data=list()
			for i in range(2):
				temp_table=[]
				for k in range(len(labels_1)):
					if labels_1[k]==values_list[i]:
						ind=new_proteins.index(proteins[j])
						if data[ind][k]!='' and data[ind][k]!=-1000:
							num_of_cat[i]+=1
							temp_table.append(data[ind][k])
				splitted_data.append(temp_table)
			mpl_fig = plt.figure()
			mpl_fig.subplots_adjust(bottom=0.15)
			ax = mpl_fig.add_subplot(111)

			ax.boxplot(splitted_data)

			ax.set_title(proteins[j])
			ax.set_ylabel('Relative Quantities')
			ax.set_xlabel(values_list[0]+' vs '+values_list[1])
			labels = [item.get_text() for item in ax.get_xticklabels()]
			labels[0] = values_list[0]+'\n(n=' +str(num_of_cat[0])+')'
			labels[1] = values_list[1]+'\n(n=' +str(num_of_cat[1])+')'

			ax.set_xticklabels(labels)
			#mpl_fig.Title(proteins[j])
			if not os.path.exists(folder):
				os.makedirs(folder)
			mpl_fig.savefig(folder+proteins[j]+'_boxplot.png')
			mpl_fig.clf()
			plt.close('all')
			#plotly_fig = tls.mpl_to_plotly( mpl_fig )
			#plot_url = py.plot(plotly_fig, 'mpl-multiple-boxplot')
	logging.info("Boxplots successfully printed!")


def average_duplicate_measurements(dataset_initial, markers):
	'''
	Average duplicate measurements.

	Args:
		dataset_initial: the initial dataset, a list of lists
		markers: input biomarkers

	Returns: a list of two lists, data (a list of lists) and markers (a single list).
	'''
	dataset={}
	dict_of_occurences={}
	num_of_elements=0
	for i in range(len(dataset_initial)):
		if dataset_initial[i][0] not in dataset:
			dict_of_occurences[markers[i]]=1
			dataset[markers[i]]=list()
			for j in range(len(dataset_initial[0])):
				if dataset_initial[i][j]!=-1000 and dataset_initial[i][j]!='':
					dataset[markers[i]].append(float(dataset_initial[i][j]))
				else:
					dataset[markers[i]].append(0)
		else:
			dict_of_occurences[markers[i]]+=1
			for j in range(len(dataset_initial[0])):
				if dataset_initial[i][j]!=-1000 and dataset_initial[i][j]!='':
					dataset[markers[i]][j]=dataset[markers[i]][j]+float(dataset_initial[i][j])
		num_of_elements+=1
	element=0
	for key in dataset:
		for j in range(len(dataset[key])):
			dataset[key][j]=dataset[key][j]/dict_of_occurences[key]
	data=list()
	markers=list()
	num_of_markers=0
	for key,vals in dataset.items():
		data.append([])
		markers.append(key)
		for i in range(len(vals)):
			data[num_of_markers].append(vals[i])
		num_of_markers+=1
	return [data,markers]


def perform_missing_values_imputation(dataset_initial, missing_imputation_method, folder_name, output_message):
	'''
	Perform missing values imputation.

	Args:
		dataset_initial: the initial dataset (a list of lists)
		missing_imputation_method (integer): 1 for average imputation, 2 for KNN-impute
		folder_name: output folder name that will hold the averages for missing values imputation
		output_message: the output message that tells which imputation method was used

	Returns: a list with the final dataset (list of lists) and the output message (string).
	'''
	#missing values imputation
	averages=[0]*(len(dataset_initial))
	if missing_imputation_method== 1:
		#average imputation here
		num_of_non_missing_values=[0]*(len(dataset_initial))
		for j in range(len(dataset_initial)):
			for i in range((len(dataset_initial[0]))):
				if dataset_initial[j][i]!=-1000 and dataset_initial[j][i]!='':
					#print(dataset_initial[j][i+1])
					averages[j]+=float(dataset_initial[j][i])
					num_of_non_missing_values[j]+=1
			averages[j]=averages[j]/float(num_of_non_missing_values[j])

		write_one_dimensional_list_to_tab_delimited_file(averages, folder_name+'averages_for_missing_values_imputation.txt')
		output_message+='Average imputation method was used!\n'
		for i in range(len(dataset_initial)):
			for j in range((len(dataset_initial[0]))):
				if dataset_initial[i][j]==-1000:
					dataset_initial[i][j]=averages[i]
		return [dataset_initial, output_message]
	else:
		#KNN-impute
		dataset_initial=list(map(list, zip(*dataset_initial)))
		for i in range(len(dataset_initial)):
			for j in range(len(dataset_initial[0])):
				if dataset_initial[i][j]=='' or dataset_initial[i][j]==-1000:
					dataset_initial[i][j]= np.NaN
		dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)), k=3)
		dataset=list(map(list, zip(*dataset)))
		output_message+='KNN imputation method was used!\n'
		return [dataset,output_message]


def normalize_dataset(dataset_initial, output_message, normalization_method):
	'''
	Normalize the dataset.

	Args:
		dataset_initial: the initial dataset (a list of lists)
		output_message: a string that holds the name of the normalization used
		normalization_method (integer): 1 for arithmetic sample-wise normalization, 2 for logarithmic normalization

	Returns: if method 1 selected a list with the normalized dataset and the output message, else for method 2
			logged data are returned along with the output message.
	'''
	if normalization_method== 1:
		#arithmetic sample-wise normalization
		maximums=[-1000.0]*(len(dataset_initial[0]))
		minimums=[1000.0]*(len(dataset_initial[0]))

		for i in range(len(dataset_initial)):
			for j in range((len(dataset_initial[0]))):
				if(dataset_initial[i][j]!='' and dataset_initial[i][j]!=-1000): # if dataset_initial == '' it blows up // changed or to and
					if float(dataset_initial[i][j])>maximums[j]:
						maximums[j]=float(dataset_initial[i][j])	# blows up for float('')
					if float(dataset_initial[i][j])<minimums[j]:
						minimums[j]=float(dataset_initial[i][j])	# blows up for float('')
		max1=max(maximums)
		min1=min(minimums)
		print('Maximum Quantity Value:'+str(max1))
		print('Minimum Quantity Value:'+str(min1))
		for i in range(len(dataset_initial)):
			for j in range((len(dataset_initial[0]))):
				if (dataset_initial[i][j]!='' and dataset_initial[i][j]!=-1000):	# if dataset_initial == '' it blows up // changed or to and
					dataset_initial[i][j]=0+(1/(max1-min1))*(float(dataset_initial[i][j])-min1)	# blows up for float('')
		output_message+='Arithmetic normalization was used!\n'
		return [dataset_initial,output_message]
	else:
		logged_data=list()
		for i in range(len(dataset_initial)):
			#print('i='+str(i))
			logged_data.append([])
			for j in range(len(dataset_initial[0])):
				#print('j='+str(j))
				if dataset_initial[i][j]=='' or dataset_initial[i][j]==-1000:
					logged_data[i].append('')
				else:
					if(dataset_initial[i][j]==0):
						logged_data[i].append(0)
					else:
						logged_data[i].append(math.log2(dataset_initial[i][j]))
		output_message+='Logarithmic normalization was used!\n'
		return[logged_data,output_message]


def print_data(data, markers, labels, folder_name, filename):
	'''
	Writes data and labels to a file.

	Args:
		data: input data (list of lists)
		markers: input biomarkers (list)
		labels: input labels (list)
		folder_name: output folder
		filename: output filename

	Returns: doesn't return anything, only writes labels and data to a file.
	'''
	file=open(folder_name+filename,'w')
	message=''
	for i in range(len(data[0])):
		message=message+'\t'+labels[i]
	message+='\n'
	for i in range(len(data)):
		message+=markers[i]
		for j in range(len(data[0])):
			message+='\t'+str(data[i][j])
		message+='\n'
	file.write(message)
	file.close()


def print_heatmap(proteins, data, labels, unique_labels, samples, scaling, filename):
	'''
	Prints heatmap image as a png.

	Args:
		proteins: list of proteins
		data: input data as a list of lists
		labels: list of labels
		unique_labels: list of unique labels
		samples: list of samples
		scaling: float
		filename: output filename

	Returns: doesn't return anything, only prints a heatmap to the output filename.
	'''
	samples_colors=list()
	for i in range(len(data[0])):
		if labels[i]==unique_labels[0]:
			samples_colors.append("#FF0000") # Red
		else:
			samples_colors.append("#0000FF") # Blue
	#print('Heatmaps as PNG')
	logging.info('Heatmaps as PNG')
	r.png(filename, width = 10, height = 10,units = 'in',res = 1200)
	data_array=np.asarray(data)
	#r.heatmap(data_array)
	r.heatmap(data_array, cexRow=scaling, labCol=np.asarray(samples),labRow=np.asarray(proteins),ColSideColors = np.asarray(samples_colors),
				col = ro.r.redgreen(75), show_heatmap_legend = True, show_annotation_legend = True)
	r("dev.off()")


def parse_data_carotid(data_filename):
	'''
	Parses data carotid.

	Args:
		data_filename: input data filename

	Returns: a list of three lists, proteins, data and samples.
	'''
	num_of_lines=0
	proteins=list()
	data=list()
	samples=list()
	with open(data_filename) as data_fname:
		for line in csv.reader(data_fname, delimiter="\t"):
			if num_of_lines==0:
				for j in range(len(line)):
					if j>0:
						samples.append(line[j].strip())
			else:
				proteins.append(line[0])
				data.append([])
				for j in range(len(line)):
					if j>0:
						if line[j]!='':
							data[num_of_lines-1].append(float(line[j]))
						else:
							data[num_of_lines-1].append(-1000)
			num_of_lines+=1
	print('Data were successfully parsed!')
	return [proteins,data,samples]


def parse_labels_carotid(labels_filename):
	'''
	Parses labels carotid.

	Args:
		labels_filename: input labels filename

	Returns: a list of two lists, symptomatic_asymptomatic, diabetes.
	'''
	symptomatic_asymptomatic=list()
	diabetes=list()
	num_of_lines=0
	with open(labels_filename) as labels_fname:
		for line in csv.reader(labels_fname, delimiter="\t"):
			if num_of_lines==0:
				for i in range(len(line)):
					if line[i].strip()=='Symptomatic':
						symptomatic_asymptomatic.append('Symptomatic')
					else:
						symptomatic_asymptomatic.append('Asymptomatic')
			else:
				for i in range(len(line)):
					diabetes.append((line[i].strip()))
			num_of_lines+=1
	print('Labels were successfully parsed!')
	return [symptomatic_asymptomatic, diabetes]


def parse_labels_new_carotid(labels_filename):
	'''
	Parses labels carotid (new).

	Args:
		labels_filename: input labels filename

	Returns: a list of the labels.
	'''
	labels=list()
	num_of_lines=0
	with open(labels_filename) as labels_fname:
		for line in csv.reader(labels_fname, delimiter="\t"):
			for i in range(len(line)):
				labels.append(line[i].strip())
	print('Labels were successfully parsed!')
	return labels


def parse_commorbidities_carotid(samples, commorbidities_filename):
	'''
	Parses commorbidities carotid.

	Args:
		samples: a list of the samples
		commorbidities_filename: the filename with the commorbidities

	Returns: a list of lists, age, sex, statin and a_or_b.
	'''
	age=list()
	sex=list()
	statin=list()
	a_or_b=list()
	commorbidities=dict()

	num_of_lines=0
	with open(commorbidities_filename) as commorbidities_fname:
		for line in csv.reader(commorbidities_fname, delimiter="\t"):
			commorbidities[line[0]]=[int(line[1].strip()), (line[2].strip()), (line[3].strip())]
			num_of_lines+=1

	for i in range(len(samples)):
		if samples[i][-1]=='A':
			a_or_b.append('A')
		else:
			a_or_b.append('B')
		age.append(commorbidities[samples[i][0:-1]][0])
		sex.append(commorbidities[samples[i][0:-1]][1])
		statin.append(commorbidities[samples[i][0:-1]][2])
	print('Commorbidities were successfully parsed!')
	return [age,sex,statin,a_or_b]


def cluster_samples(samples, data):
	'''
	Cluster samples.

	Args:
		samples: input samples, list
		data: input data in the form [samples X PCAs]

	Returns: list of labels.
	'''
	data = StandardScaler().fit_transform(data)
	print(data)
	#data should be [samples X PCAs]
	db = DBSCAN(eps=0.3, min_samples=3).fit(data)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	print(labels)
	print(len(labels))
	print(samples)
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)

	#print("Silhouette Coefficient: %0.3f"		  % metrics.silhouette_score(data_for_clustering, labels))
	return labels


def wilcoxon_rank_sum_test(proteins, control, condition, paired_flag ):
	'''
	Wilcoxon rank sum test.

	Args:
		proteins: a list with the proteins
		control: a list of lists
		condition: a list of lists
		paired_flag: if zero it uses the Ranksums, if not it uses Wilcoxon.

	Returns: a list of the lists pvals, folds and stdevs.
	'''
	pvals=list()
	Zs=list()
	upregulated_proteins=list()
	downregulated_proteins=list()
	num_of_ups=0
	num_of_downs=0
	folds=list()
	stdevs=list()
	for i in range(len(proteins)):
		control_data_per_protein=list()
		condition_data_per_protein=list()
		for j in range(len(control[i])):
			control_data_per_protein.append(control[i][j])
		for j in range(len(condition[i])):
			condition_data_per_protein.append(condition[i][j])

		#[z,pval]=st.wilcoxon(control_data_per_protein, condition_data_per_protein)
		if statistics.stdev(control_data_per_protein)==0 and statistics.stdev(condition_data_per_protein)==0 and statistics.mean(control_data_per_protein)==statistics.mean(condition_data_per_protein):
			pval=1
		else:
			if paired_flag==0:
				[z,pval]=st.ranksums(control_data_per_protein, condition_data_per_protein)
			else:
				[z,pval]=st.wilcoxon(control_data_per_protein, condition_data_per_protein)
		pvals.append(pval)

		if paired_flag==1:
			fold_per_sample=list()
			for k in range(len(condition_data_per_protein)):
				fold_per_sample.append(condition_data_per_protein[k]-control_data_per_protein[k])
		if statistics.mean(control_data_per_protein)==0:
			folds.append(0)
			stdevs.append(0)
		else:
			folds.append(statistics.mean(condition_data_per_protein)-statistics.mean(control_data_per_protein))
			if paired_flag==1:
				stdevs.append(statistics.stdev(fold_per_sample))
			else:
				stdevs.append(0)
	if paired_flag==0:
		stdevs=[0]*len(proteins)
	logging.info("Wilcoxon rank sum test successfully finished!")
	return [pvals,folds,stdevs]


def print_all_plots(proteins1, proteins2, pvals, pval2, log_fold_changes, samples,
						filename_volcano_corrected,
						filename_volcano,
						p_value_threshold, dataset_imputed, unique_labels, labels,
						filename_boxplots_corrected,
						filename_boxplots,
						filename_heatmap_not_corrected,
						filename_heatmap_corrected,
						filename_heatmap_all, folder_name,
						parametric_test_flag):
	'''
	Prints all plots that are called from the main function, run_statistical_analysis.

	Args:
		proteins1: list of proteins 1
		proteins2: list of proteins2
		pvals: list of pvals
		pval2: list of pval 2
		log_fold_changes: log fold changes list
		samples: list of samples
		filename_volcano_corrected: filename of volcano corrected file
		filename_volcano: filename of volcano file
		p_value_threshold: p value threshold
		dataset_imputed: imputed dataset, list of lists
		unique_labels: unique labels, list
		labels: labels list
		filename_boxplots_corrected: filename of boxplots corrected file
		filename_boxplots: filename of boxplots file
		filename_heatmap_not_corrected: filename of heatmap not corrected file
		filename_heatmap_corrected: filename of heatmap corrected file
		filename_heatmap_all: filename of heatmap_all file
		parametric_test_flag: parametric test flag, 1 for parametric test, 2 for non parametric test

	Returns: nothing, it just calls the plotting functions.
	'''

	if parametric_test_flag == 1:
		protein = proteins1 #diff_proteins
		p_value = pvals # diff_table[3]
		p_value2 = pval2 # diff_table[4]
	else:
		protein = proteins2 #markers
		p_value = pvals  # pvals
		p_value2 = pval2  # pvals2

	print_volcano_plots(protein, p_value2, log_fold_changes, filename_volcano_corrected, p_value_threshold)
	print_volcano_plots(protein, p_value, log_fold_changes, filename_volcano, p_value_threshold)
	#print_boxplots(proteins2, protein, dataset_imputed, unique_labels, labels, filename_boxplots_corrected, pvals, p_value_threshold)
	#print_boxplots(proteins2, protein, dataset_imputed, unique_labels, labels, filename_boxplots, pval2, p_value_threshold)
	tags = list(set(labels))
	dataset_imputed = np.transpose(dataset_imputed)
	print_significant_boxplots(protein, dataset_imputed, labels, folder_name+'boxplots/', 0, pvals, tags, p_value_threshold)
	dataset_imputed = np.transpose(dataset_imputed)
	#print_significant_data(dataset_imputed, proteins2, samples, folder_name)
	try:
		print_significant_data_better(dataset_imputed, protein, proteins2, samples, pvals, p_value_threshold, folder_name)
	except Exception as e:
		logging.exception("Exception during printing of significant molecules file.")

	filtered_data = list()
	position = 0
	diff_proteins_corrected = list()
	for i in range(len(protein)):
		prot = protein[i]
		if p_value[i] < p_value_threshold:
			diff_proteins_corrected.append(prot)
			ind = proteins2.index(prot)
			filtered_data.append([])

			for j in range(len(dataset_imputed[ind])):
				filtered_data[position].append(dataset_imputed[ind][j])
			position += 1
	if position > 1:
		try:
			print_heatmap(diff_proteins_corrected, filtered_data, labels, unique_labels, samples, 0.4, filename_heatmap_not_corrected)
		except Exception as e:
			logging.exception("Exception during printing of heatmap not corrected.")

	filtered_data = list()
	position = 0
	diff_proteins_corrected = list()
	for i in range(len(protein)):
		prot = protein[i]
		if p_value2[i] < p_value_threshold:
			diff_proteins_corrected.append(prot)
			ind = proteins2.index(prot)
			filtered_data.append([])

			for j in range(len(dataset_imputed[ind])):
				filtered_data[position].append(dataset_imputed[ind][j])
			position += 1
	if position > 1:
		try:
			print_heatmap(diff_proteins_corrected,filtered_data, labels, unique_labels, samples, 0.4, filename_heatmap_corrected)
		except Exception as e:
			logging.exception("Exception during printing of heatmap corrected.")
	try:
		print_heatmap(proteins2, dataset_imputed,labels, unique_labels, samples, 0.4, filename_heatmap_all)
	except Exception as e:
		logging.exception("Exception during printing of heatmap all.")
	logging.info("Plots successfully printed!")


def print_significant_boxplots(proteins, data, annotation, folder, folded, pvals, tags, p_value_threshold):
	'''
	Prints significant boxplots.

	Args:
		proteins: a list with the proteins
		data: the input data, a list of lists
		annotation: a list with the annotations
		folder: the output folder
		folded: if folded is equal to 1, we have Folded Quantites on the y axis, else we have Logged Relative Quantities
		pvals: a list with the p-values
		tags: a list with the tags
		p_value_threshold: the p value threshold, float

	Returns: nothing, just prints the significant boxplots into the given output folder.
	'''
	for j in range(len(proteins)):
		if pvals[j] < float(p_value_threshold):
			splitted_data = list()
			for i in range(len(tags)):
				temp_table = []
				for k in range(len(data)):
					if annotation[k] == tags[i]:
						temp_table.append(data[k][j])
				splitted_data.append(temp_table)
			mpl_fig = plt.figure(figsize=((len(tags)+1), 5))
			mpl_fig.subplots_adjust(bottom=0.25, left=0.2)
			ax = mpl_fig.add_subplot(111)
			ax.boxplot(splitted_data, widths=0.02, positions= (np.arange(len(tags))+0.5)/(len(tags)+1) )
			#ax.set_aspect(1.2)
			ax.set_title(proteins[j])
			if folded == 1:
				ax.set_ylabel('Folded Quantities')
			else:
				ax.set_ylabel('Logged Relative Quantities')
			ax.set_xlabel('Phenotypes')
			labels = [item.get_text() for item in ax.get_xticklabels()]
			for i in range(len(tags)):
				labels[i] = tags[i]
			ax.set_xticklabels(labels, rotation=45)
			ax.set_xticks((np.arange(len(labels))+0.5)/(len(tags)+1))
			ax.set_xlim(right=0.9, left=0)
			if not os.path.exists(folder):
				os.makedirs(folder)
			mpl_fig.savefig(folder+proteins[j]+'_boxplot.png')
			mpl_fig.clf()
			plt.close('all')
	logging.info("Significant boxplots successfully printed!")

def print_data2(data, markers, labels, folder_name, filename):
	'''
	Writes data and labels to a file.

	Args:
		data: input data (list of lists)
		markers: input biomarkers (list)
		labels: input labels, sample names (list)
		folder_name: output folder
		filename: output filename

	Returns: doesn't return anything, only writes markers, sample names and data to a file.
	'''
	file = open(folder_name+filename,'w')
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

def print_data_nosamples(data, markers, folder_name, filename):
	'''
	Writes data and labels to a file.

	Args:
		data: input data (list of lists)
		markers: input biomarkers (list)
		folder_name: output folder
		filename: output filename

	Returns: doesn't return anything, only writes markers, sample names and data to a file.
	'''
	file=open(folder_name+filename,'w')
	message=''
	for i in range(len(data)):
		message+=markers[i]
		for j in range(len(data[0])):
			message+='\t'+str(data[i][j])
		message+='\n'
	file.write(message)
	file.close()

def print_significant_data_better(data, markers1, markers2,  samples, pvals, p_value_threshold, output_folder):
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
	for j in range(len(markers1)):
		if pvals[j] < float(p_value_threshold):
			new_proteins.append(markers1[j])
	indexes = [markers2.index(new_proteins[i]) for i in range(len(new_proteins))]
	new_data = [data[i] for i in indexes]
	if len(new_data) == 0:
		raise ValueError("The list new_data is empty, problem with pvalue threshold.")
	else:
		print_data2(new_data, new_proteins, samples, output_folder, "significant_molecules_dataset.tsv")


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
		#new_samples = list()
		for i,label in enumerate(labels):
			if tag == label:
				new_dataset.append(data_transp[i])
				#new_samples.append(samples[i])
		new_dataset = np.transpose(new_dataset)
		print_data_nosamples(new_dataset, markers, output_folder, output_filename + tag + ".tsv")


def create_differential_expression_file(data1:list, data2:list, gene_names:list, logged_flag:int, sign_threshold:float, method:int,
										output_folder:str, filename_result:str):
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

def new_parse_data(data_filename,delimiter):
	'''
	Parses data.

	Args:
		data_filename (string): dataset filename
		delimiter (string): the kind of delimiter with values "," or "\t"

	Returns: a list of three lists, [proteins, data].
	'''
	num_of_lines=0
	proteins=list()
	data=list()
	with open(data_filename) as data_fname:
		for line in csv.reader(data_fname, delimiter=delimiter):
			proteins.append(line[0])
			data.append([])
			for j in range(len(line)):
				if j>0:
					if line[j]!='':
						data[num_of_lines].append(float(line[j]))
					else:
						data[num_of_lines].append('')
			num_of_lines+=1
	#print('Data were successfully parsed!')
	logging.info('Data were successfully parsed!')
	return [proteins,data]

def run_statistical_analysis(proteomics_data, labels, markers, samples, commorbidities_filename, commorbidities_types_filename, parametric_flag,
	p_value_threshold, paired_flag, logged_flag, folder_name):
	'''
	Run statistical analysis.

	Args:
		proteomics_data: the proteomics data (list of lists), where each column is a sample and each line is a proteom. This means that it takes the data inverted from before.
		labels: the labels list
		markers (list): the names of the markers
		samples (list): the names of the samples
		commorbidities_filename: the commorbidities filename
		commorbidities_types_filename: the commorbidities types filename
		parametric_flag (integer): parametric flag, 0 if you don't know if you want parametric or non-parametric testing to be applied, 1  if you wish parametric testing to be applied
						, 2 if you wish non-parametric testing to be applied
		p_value_threshold: p value threshold
		paired_flag: needed for the parametric test and for wilkoxon rank sum test
		logged_flag: 0 if data are not logged, 1 if data are
		folder_name: output folder name

	Returns: a list, [1, string1] for successfull operation and [0, string2] for unsuccessfull operation.
	'''

	try:
		# Parsing
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)

		[commorbidities_types, commorbidities_flag] = parse_commorbidities_types(commorbidities_types_filename)

		unique_labels = list(set(labels))

		# Parse commorbidities
		if commorbidities_flag != 0:
			[commorbidities, commorbidities_flag] = parse_commorbidities(commorbidities_filename)
		else:
			commorbidities=list()

		output_message = ''
		if commorbidities_flag == 0:
			output_message += 'No commorbidities were provided!\n'
			logging.info("No commorbidities were provided!")
		else:
			output_message += str(len(commorbidities))+' commorbidities were provided.\n'
			logging.info(str(len(commorbidities))+' commorbidities were provided.')

	except	Exception as e:
		logging.exception("Statistical analysis raised the exception during parsing.")
		return [0, "Statistical analysis raised the exception during parsing: {}".format(str(e))]


	try:
		dataset_imputed = proteomics_data

		# Zero if you don't know if you want parametric or non-parametric testing to be applied
		if parametric_flag== 0:
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
				logging.info('Data are normally distributed. Parametric testing will be done.')
			else:
				test_flag = 2
				output_message += 'Data are not normally distributed. Non parametric testing will be done.\n'
				logging.info('Data are not normally distributed. Non parametric testing will be done.')
		# One if you wish parametric testing to be applied
		elif parametric_flag == 1:
			test_flag = 1
			output_message += 'Parametric testing will be done.\n'
			logging.info('Parametric testing will be done.')
		# Two if you wish non-parametric testing to be applied
		else:
			test_flag = 2
			output_message += 'Non Parametric testing will be done.\n'
			logging.info('Non Parametric testing will be done.')


		# Begin statistical tests
		if test_flag == 1:
			# Begin parametric test
			logging.info("Begin parametric test")
			if paired_flag == 1:
				counter1 = 0
				counter2 = 0
				new_com = list()
				for j in range(len(dataset_imputed[0])):
					if labels[j] == unique_labels[0]:
						new_com.append(counter1)
						counter1 += 1
					else:
						new_com.append(counter2)
						counter2 += 2
				commorbidities_flag = 1
				commorbidities_types = list()
				commorbidities_types.append('0')

			# Differential expression analysis
			#[diff_table, diff_proteins, diff_columns]=differential_expression_analysis_new(markers, dataset_imputed, labels,commorbidities_flag,
			#									commorbidities, commorbidities_types, unique_labels, folder_name, unique_labels[0]+'VS'+unique_labels[1])

			[diff_table, diff_proteins, diff_columns] = differential_expression_analysis_new(markers, dataset_imputed, labels, commorbidities_flag,
															commorbidities, commorbidities_types, folder_name)

			# Print plots
			print_all_plots(diff_proteins, markers, diff_table[diff_columns.index('P.Value')], diff_table[diff_columns.index('adj.P.Val')], diff_table[diff_columns.index('logFC')], samples,
						folder_name+'volcano_plot_corrected.png',
						folder_name+'volcano_plot.png',
						p_value_threshold, dataset_imputed, unique_labels, labels,
						folder_name+'boxplots_corrected/',
						folder_name+'boxplots_uncorrected/',
						folder_name+'heatmap_significant_not_corrected.png',
						folder_name+'heatmap_significant_corrected.png',
						folder_name+'heatmap_all.png', folder_name,
						parametric_test_flag=test_flag)
			logging.info("Parametric test finished.")
		else:
			# Begin non parametric test
			logging.info("Begin non parametric test.")
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
			try:
				[pvals, folds, stdevs] = wilcoxon_rank_sum_test(markers, category1, category2, paired_flag )
			except Exception as e:
				logging.exception("Statistical analysis raised the exception wilcoxon rank sum test. Please use unpaired analysis.")
				return [0, "Statistical analysis raised the exception during wilcoxon rank sum test: {} Please use unpaired analysis.".format(str(e))]
			for k in range(len(pvals)):
				if 'nan' in str(pvals[k]):
					pvals[k] = 1
			pvalues = smm.multipletests(pvals, method='fdr_bh')
			pvals2 = pvalues[1]

			header = 'IDs\tInitial Pvalue\tAdjusted Pvalue\tFold Change\tStandard Deviation of Fold Changes\n'
			all_pvals_data_zipped = zip(markers, pvals, pvals2, folds, stdevs)
			all_pvals_data_zipped = list(all_pvals_data_zipped)
			all_pvals_data_zipped.sort(key=lambda x:x[1], reverse=False)
			with open(folder_name + "all_pvals.tsv","w") as handle:
				handle.write(header)
				for id, init_pval, adj_pval, fold_change, stdev in all_pvals_data_zipped:
					handle.write(str(id) + "\t"
								 + str(init_pval) + "\t"
								 + str(adj_pval) + "\t"
								 + str(fold_change) + "\t"
								 + str(stdev) + "\n")

			os.mkdir(folder_name + "top20/")
			with open(folder_name + "top20/" + "all_pvals_top20.tsv","w") as handle:
				handle.write(header)
				for i, (id, init_pval, adj_pval, fold_change, stdev) in enumerate(all_pvals_data_zipped):
					if i < 20:
						handle.write(str(id) + "\t"
								 + str(init_pval) + "\t"
								 + str(adj_pval) + "\t"
								 + str(fold_change) + "\t"
								 + str(stdev) + "\n")

			# Print plots
			print_all_plots(markers, markers, pvals, pvals2, folds, samples,
						folder_name+'volcano_plot_corrected.png',
						folder_name+'volcano_plot.png',
						p_value_threshold, dataset_imputed, unique_labels, labels,
						folder_name+'boxplots_corrected/',
						folder_name+'boxplots_uncorrected/',
						folder_name+'heatmap_significant_not_corrected.png',
						folder_name+'heatmap_significant_corrected.png',
						folder_name+'heatmap_all.png', folder_name,
						parametric_test_flag=test_flag)

			logging.info("Non parametric test finished!")
		out_file = open(folder_name+'info.txt','w')
		out_file.write(output_message)
		out_file.close()
	except Exception as e:
		logging.exception("Statistical analysis raised the exception during testing.")
		if str(e) == "Error in .ebayes(fit = fit, proportion = proportion, stdev.coef.lim = stdev.coef.lim,  : \n  No residual degrees of freedom in linear model fits\n":
			return [0, "{}. Please use non parametric method !".format(str(e))]
		else:
			return [0, "Statistical analysis raised the exception during testing: {}".format(str(e))]

	try:
		# Printing the MQ files for the full dataset
		os.mkdir(folder_name + "MQ_files")
		create_MQ_files(dataset_imputed, markers, labels, "MQ_", "{0}MQ_files/".format(folder_name))

		# First parsing the significant  molecules dataset
		filename = folder_name + "significant_molecules_dataset.tsv"
		if os.path.isfile(filename):
			os.mkdir(folder_name + "MQ_significant_files")
			sign_markers, sign_dataset, samples = parse_data(filename)
			# Printing the MQ files for the significant dataset
			create_MQ_files(sign_dataset, sign_markers, labels, "MQ_significant_",
							"{0}MQ_significant_files/".format(folder_name))
	except Exception as e:
		logging.exception("Statistical analysis raised an exception during creation of MQ files.")
		return [0, "Statistical analysis raised an exception during creation of MQ files: {}. ".format(str(e))]

	try:
		if os.path.isfile(filename):
			# Creating the differential expression file
			os.mkdir(folder_name + "diff_expression_files")
			input_files = os.listdir(folder_name + "MQ_significant_files/")
			proteins1, dataset1 = new_parse_data(folder_name + "MQ_significant_files/"+ input_files[0], "\t")
			proteins2, dataset2 = new_parse_data(folder_name + "MQ_significant_files/"+ input_files[1], "\t")
			create_differential_expression_file(dataset1, dataset2, proteins1, logged_flag, p_value_threshold, test_flag,
												folder_name + "diff_expression_files/",
												"diff_express_file_{0}_VS_{1}.tsv".format(input_files[0][:-4],
																						  input_files[1][:-4]))
	except Exception as e:
		logging.exception("Statistical analysis raised an exception during creation of diff. expression file.")
		return [0, "Statistical analysis raised an exception during creation of diff. expression file: {}.".format(str(e))]

	return [1, "Job completed successfully."]


if __name__ == "__main__":
	print(0)
	#Example run: python statistical_analysis_v2_0.py example_dataset.txt example_labels.txt commorbidities2.txt commorbidities_types.txt 0 2 2 0.1 0.05 0 statistical_analysis_results2/

	#result = run_statistical_analysis("example_dataset.txt", "example_labels.txt","commorbidities2.txt", "commorbidities_types.txt",1, 2, 2, 0.1, 0.05, 0,
	#									"statistical_analysis_results8/", "my_preprocessed_data.txt")
	#result = run_statistical_analysis("example_dataset.txt", "example_labels.txt","commorbidities2.txt", "commorbidities_types_new.txt",1, 2, 2, 0.1, 0.05, 0,
	#									"statistical_analysis_results9/", "my_preprocessed_data_new.txt")
	#result = run_statistical_analysis("testing/training_data_trunc.txt", "testing/training_labels.txt","commorbidities2.txt", "commorbidities_types_new.txt",1, 2, 2, 0.1, 0.05, 0,
	#									"statistical_analysis_results10/", "my_preprocessed_data_new.txt")
	#result = run_statistical_analysis("testing/test_data_truncated.txt", "testing/test_labels.txt","commorbidities2.txt", "commorbidities_types_new.txt",1, 2, 2, 0.1, 0.05, 0,
	#									"statistical_analysis_results10/", "my_preprocessed_test_data_new.txt")
	#print(result)


