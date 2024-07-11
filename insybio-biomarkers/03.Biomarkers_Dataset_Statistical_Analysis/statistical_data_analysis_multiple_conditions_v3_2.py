
#python statistical_data_analysis_timeseries.py data_example.txt 4 tags.txt
#python statistical_data_analysis_timeseries.py miRNA_protein_data.txt 12 miRNA_protein_tags.txt
#python statistical_data_analysis_timeseries.py tissue_crhistian_data.txt 6 christian_labels.txt
import matplotlib
matplotlib.use('Agg')
import os
import itertools
import statistics
import numpy as np
import scipy.stats as st
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smm
import plotly
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

os.environ['R_HOME'] = "/usr/lib/R"
os.environ['R_USER'] = '/usr/lib/python3/dist-packages/rpy2/'

from rpy2 import *
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
ro.numpy2ri.activate()
from rpy2.robjects import r
import logging


gplots = importr("gplots")
limma = importr('limma') 
statmod=importr('statmod')
lattice=importr('lattice')


def parse_data(data_filename):
	'''
	Parses data.

	Args:
		data_filename: input data filename

	Returns: a list with the proteins and the data.
	'''
	num_of_lines=0
	proteins=list()
	data=list()
	with open(data_filename) as data_fname:
			for line in csv.reader(data_fname, delimiter="\t"):
				if num_of_lines==0:
					for j in range(len(line)):
						proteins.append(line[j].strip().replace(" ", "").replace('/',''))
				else:
					
					data.append([])
					for j in range(len(line)):
						if line[j]!='' and line[j]!=-1000:
							data[num_of_lines-1].append(float(line[j]))
						elif line[j]!='':
							data[num_of_lines-1].append(-1000)
						else:
							data[num_of_lines-1].append('')
				num_of_lines+=1
	#print('Data were successfully parsed!')
	logging.info('Data were successfully parsed inside stat data analysis multiple conditions!')
	return [proteins,data]

def parse_data_all_headers(data_filename):
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


def parse_samples(samples_filename):
	'''
	Parses the samples.

	Args:
		samples_filename:	the samples filename

	Returns: a list with the samples.
	'''
	samples=list()
	num_of_lines=0
	with open(samples_filename) as samples_fname:
		for line in csv.reader(samples_fname, delimiter="\t"):
			if num_of_lines>0:
				samples.append(line[1])
			num_of_lines+=1
	logging.info("Samples successfully parsed!")
	return samples

def parse_samples2(samples_filename):
	'''
	Parses the samples.

	Args:
		samples_filename:	the samples filename

	Returns: a list with the samples.
	'''
	samples=list()
	with open(samples_filename) as samples_fname:
		for line in samples_fname:
			words=line.split('\t')
			for sample in words:
				samples.append(sample.strip())
	logging.info("Samples successfully parsed!")
	return samples			


def parse_labels(tags_filename):
	'''
	Parses the labels filename.

	Args:
		tags_filename: the labels filename, one line tab seperated values

	Returns:
		labels (list): the list of labels
	'''
	labels=list()
	with open(tags_filename) as tags_fname:
		for line in tags_fname:
			words=line.split('\t')
			for label in words:
				labels.append(label.strip())
	logging.info("Labels successfully parsed inside stat. data analysis multiple conditions!")
	return labels
	
def filter_data(data, samples):
	'''
	Filters the data.

	Args:
		data: input data
		samples: sampled data
	
	Returns: a list with the filtered data.
	'''
	filtered_data=list()
	for sample in samples:
		filtered_data.append(data[sample])
	logging.info("Data successfully filtered!")
	return filtered_data


def fold_data(data, timepoints):
	'''
	Folds data.

	Args:
		data: input data
		timepoints: the number of timepoints

	Returns: a list with new data.
	'''
	data_new=copy.deepcopy(data)
	for j in range(len(data_new[0])):
		for i in range(len(data_new)):
			if i%timepoints==0:
				for k in range(timepoints):
					if data_new[i][j]==0:
						data_new[i+timepoints-i][j]=0
					else:
						data_new[i+timepoints-i][j]/=data_new[i][j]
	return data_new


def data_per_timepoints(data, timepoints):
	'''
	Calculates the data per timepoints.

	Args:
		data: input data
		timepoints: the number of timepoints

	Returns: a list with the splitted data.
	'''
	splitted_data=list()
	for i in range(timepoints):
		temp_table=[]
		for j in range(len(data)):
			if j%timepoints==i:
				temp_table.append(data[j])
		splitted_data.append(temp_table)		
	return splitted_data

	
def print_boxplots(proteins, data, annotation, folder, folded, tags):
	'''
	Prints boxplots.

	Args:
		proteins: a list with the proteins
		data: the input data, a list of lists
		annotation: a list with the annotations
		folder: the output folder
		folded: if folded is equal to 1, we have Folded Quantites on the y axis, else we have Logged Relative Quantities
		tags: a list with the tags

	Returns: nothing, just prints the boxplots inside the given folder.
	'''
	for j in range(len(proteins)):	
		splitted_data=list()
		for i in range(len(tags)):
			temp_table=[]
			for k in range(len(data)):
				if annotation[k]==tags[i]:
					temp_table.append(data[k][j])
			splitted_data.append(temp_table)	
		mpl_fig = plt.figure(figsize=((len(tags)+1), 5))
		mpl_fig.subplots_adjust(bottom=0.25,left=0.2)
		ax = mpl_fig.add_subplot(111)
		ax.boxplot(splitted_data, widths=0.02, positions= (np.arange(len(tags))+0.5)/(len(tags)+1) )
		#ax.set_aspect(1.2)
		ax.set_title(proteins[j])
		if folded==1:
			ax.set_ylabel('Folded Quantities')
		else:
			ax.set_ylabel('Logged Relative Quantities')
		ax.set_xlabel('Phenotypes')
		labels = [item.get_text() for item in ax.get_xticklabels()]
		for i in range(len(tags)):
			labels[i]=tags[i]
		ax.set_xticklabels(labels, rotation=45)
		ax.set_xticks((np.arange(len(labels))+0.5)/(len(tags)+1))
		ax.set_xlim(right=0.9, left=0)
		if not os.path.exists(folder):
			os.makedirs(folder)
		mpl_fig.savefig(folder+proteins[j]+'_boxplot.png')
		mpl_fig.clf()
		plt.close('all')
	logging.info("Boxplots successfully printed")
		

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
		if(pvals[j]<float(p_value_threshold)):
			splitted_data=list()
			for i in range(len(tags)):
				temp_table=[]
				for k in range(len(data)):
					if annotation[k]==tags[i]:
						temp_table.append(data[k][j])
				splitted_data.append(temp_table)	
			mpl_fig = plt.figure(figsize=((len(tags)+1), 5))
			mpl_fig.subplots_adjust(bottom=0.25,left=0.2)
			ax = mpl_fig.add_subplot(111)
			ax.boxplot(splitted_data, widths=0.02, positions= (np.arange(len(tags))+0.5)/(len(tags)+1) )
			#ax.set_aspect(1.2)
			ax.set_title(proteins[j])
			if folded==1:
				ax.set_ylabel('Folded Quantities')
			else:
				ax.set_ylabel('Logged Relative Quantities')
			ax.set_xlabel('Phenotypes')
			labels = [item.get_text() for item in ax.get_xticklabels()]
			for i in range(len(tags)):
				labels[i]=tags[i]
			ax.set_xticklabels(labels, rotation=45)
			ax.set_xticks((np.arange(len(labels))+0.5)/(len(tags)+1))
			ax.set_xlim(right=0.9, left=0)
			if not os.path.exists(folder):
				os.makedirs(folder)
			mpl_fig.savefig(folder+proteins[j]+'_boxplot.png')
			mpl_fig.clf()
			plt.close('all')
	logging.info("Significant Boxplots successfully printed")		


def kruskal_wallis_test(data, labels):
	'''
	Performs the Kruskal-Wallis statistical test.

	Args:
		data: the input data, a list of lists
		labels: the list of labels		

	Returns: a list with two lists, the values of H (Hs) and the values of p (pvals).
	'''
	Hs=list()
	pvals=list()
	tags=list(set(labels))	
	#tags=['Fibroatheroma','Complex','Calcified','Fibrotic']
	#tags=['A','C','D','v1H','v2H']	
	for j in range(len(data[0])):
		splitted_data=dict()
		for i in range(len(tags)):
			splitted_data[i]=list()
			for k in range(len(data)):
				if labels[k]==tags[i]:
					splitted_data[i].append(data[k][j])		
		try:
			H, pval =mstats.kruskalwallis(*splitted_data.values())			
		except:
			H=0
			pval=1
		Hs.append(H)
		pvals.append(pval)
	logging.info("Kruskal wallis test successfully finished!")
	return [Hs,pvals]

	
def anova_test(data, labels):
	'''
	Performs the Anova test.

	Args:
		data: the input data, a list of lists
		labels: list of labels		

	Returns: a list with two lists, the values of H (Hs) and the values of p (pvals).
	'''
	Hs=list()
	pvals=list()
	tags=list(set(labels))	
	#tags=['Fibroatheroma','Complex','Calcified','Fibrotic']
	#tags=['A','C','D','v1H','v2H']	
	for j in range(len(data[0])):
		splitted_data=dict()				
		for i in range(len(tags)):
			splitted_data[i]=list()
			for k in range(len(data)):
				if labels[k]==tags[i]:
					splitted_data[i].append(data[k][j])		
		try:
			H, pval =mstats.f_oneway(*splitted_data.values())		
		except:
			H=0
			pval=1
		Hs.append(H)
		pvals.append(pval)
	logging.info("Anova test successfully finished !")
	return [Hs,pvals]
	

def create_clustering_dataset(proteins, data, filename):
	'''
	Creates the clustering dataset.

	Args:
		proteins: the list of proteins
		data: a list of lists, the input data
		filename: the output filename

	Returns: nothing, it just writes the data for clustering into a csv.
	'''
	data_for_clustering=list()
	for j in range(len(proteins)):
		data_for_clustering.append([])
		data_for_clustering[j].append(proteins[j])
		splitted_data=list()
		for i in range(timepoints):
			temp_table=[]
			for k in range(len(data)):
				if k%timepoints==i:
					temp_table.append(data[k][j])
			data_for_clustering[j].append(sum(temp_table)/float(len(temp_table)))
		splitted_data.append(temp_table)
	with open(filename,'wb') as resultFile:
		wr = csv.writer(resultFile, dialect='excel')
		wr.writerows(data_for_clustering)
		

def cluster_proteins(proteins, data, filename, name):
	'''
	Creates clusters of proteins.

	Args:
		proteins: the input proteins, a list
		data: the input data, a list of lists
		filename: the name of the output filename with the created clusters
		name: a name for the clustering images

	Returns: nothing, writes the clustered clusters into a filename and creates clustering images
	'''
	data_for_clustering=list()
	for j in range(len(proteins)):
		data_for_clustering.append([])
		#data_for_clustering[j].append(proteins[j])
		splitted_data=list()
		for i in range(timepoints):
			temp_table=[]
			for k in range(len(data)):
				if k%timepoints==i:
					temp_table.append(data[k][j])
			data_for_clustering[j].append(sum(temp_table)/float(len(temp_table)))
		splitted_data.append(temp_table)
	db = DBSCAN(eps=1, min_samples=2).fit(data_for_clustering)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	print(labels)
	print(len(labels))
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)
	
	print("Silhouette Coefficient: %0.3f"
		  % metrics.silhouette_score(data_for_clustering, labels))	

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	print(unique_labels)
	colors = [plt.cm.Spectral(each)
			  for each in np.linspace(0, 1, len(unique_labels))]
	resultFile= open(filename,'w') 
	message=''
	for k, col in zip(unique_labels, colors):		
		class_member_mask = (labels == k)
		data_for_clustering_filtered=list()
		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)
		protein_list=''
		for iter in range(len(data_for_clustering)):
			#print(labels)
			#print(iter)
			if labels[iter]==k:
				#data_for_clustering_filtered.append(copy.deepcopy(data_for_clustering_filtered))
				protein_list=protein_list+'\t'+proteins[iter]
				ax.plot(data_for_clustering[iter])							
		ax.set_title('Cluster_'+str(k))
		ax.set_ylabel('Folded Concetrations')
		ax.set_xlabel('Time-points')
		#labels_n=list()
		labels_n = [item.get_text() for item in ax.get_xticklabels()]		
		labels_n [1] = '0'
		labels_n [3] ='60min'
		labels_n [5] = '8h'
		labels_n [7] ='24h'
		ax.set_xticklabels(labels_n)
		#mpl_fig.Title(proteins[j])
		mpl_fig.savefig(name+'_Cluster_'+str(k)+'_figure.png')
		mpl_fig.clf()
		
		message+='Cluster_'+str(k)+':'+protein_list+'\n'
		#
	resultFile.write(message)
	resultFile.close()	


def wilcoxon_rank_sum_test(proteins, control, condition, paired_flag ):
	'''
	Performs Wilcoxon or rank sum test.

	Args:
		proteins: the list with the proteins
		control: the control data, a list of lists
		condition: the condition data, a list of lists
		paired_flag (integer): if 0 does Ranksums else does Wilcoxon test.

	Returns: a list with the p-values (pvals), the folds and standard deviations (stdevs).
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
		for j in range(len(control)):
			control_data_per_protein.append(control[j][i])
		for j in range(len(condition)):
			condition_data_per_protein.append(condition[j][i])
		
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
	

def t_test(proteins, control, condition, paired_flag ):
	'''
	Performs the t-test.

	Args:
		proteins: the list with the proteins
		control: the control data, a list of lists
		condition: the condition data, a list of lists
		paired_flag: if 1 calculates the T-test on TWO RELATED samples of scores (ttest_rel), 
					 else it calculates the T-test for the means of TWO INDEPENDENT samples of scores (ttest_ind).

	Returns: a list with the p-values (pvals), the folds and standard deviations (stdevs).
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
		for j in range(len(control)):
			control_data_per_protein.append(control[j][i])
		for j in range(len(condition)):
			condition_data_per_protein.append(condition[j][i])
				
		if statistics.stdev(control_data_per_protein)==0 and statistics.stdev(condition_data_per_protein)==0 and statistics.mean(control_data_per_protein)==statistics.mean(condition_data_per_protein):
			pval=1
		else:
			if paired_flag==1:
				[z,pval]=st.ttest_rel(control_data_per_protein, condition_data_per_protein)
			else:
				[z,pval]=st.ttest_ind(control_data_per_protein, condition_data_per_protein)
			
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
	logging.info("T-test successfully finished!")
	return [pvals,folds,stdevs]


def perform_pairwise_analysis(proteins, data, name_prefix, annotation, tags, parametric_flag, p_value_threshold, output_message, paired_flag):
	'''
	Performs pairwise analysis.

	Args:
		proteins: list of proteins
		data: list of input data
		name_prefix: name prefix for the output file
		annotation: annotation list
		tags: list with tags 
		parametric_flag (integer): if 0 or 2, Wilcoxon test was used, else t-test was used for pairwise comparisons
		p_value_threshold: p value threshold
		output_message: the message which informs which methods of pairwise comparisons have been used
		paired_flag (integer): paired-flag flag needed for wilcoxon_rank_sum_test() or t_test() functions

	Returns: output_message
	'''	
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

	counter = 0
	for i in range(len(tags)):
		for j in range(len(tags)):
			if j > i:
				if parametric_flag==0 or parametric_flag==2:
					if counter < 1:
						output_message+='Wilcoxon test was used for pairwise comparisons\n'
						logging.info('Wilcoxon test was used for pairwise comparisons!')
						counter += 1
					[pvals, folds,stdevs]=wilcoxon_rank_sum_test(proteins,splitted_data[i], splitted_data[j],paired_flag )
				else:
					[pvals, folds,stdevs]=t_test(proteins,splitted_data[i], splitted_data[j], paired_flag )
					if counter < 1:
						output_message+='Paired t-test was used for pairwise comparisons\n'
						logging.info('Paired t-test was used for pairwise comparisons!')
						counter += 1
				for k in range(len(pvals)):
					if 'nan' in str(pvals[k]):
						pvals[k]=1
				pvalues=smm.multipletests(pvals,method='fdr_bh')
				pvals2=pvalues[1]

				header = 'IDs\tInitial Pvalue\tAdjusted Pvalue\tFold Change\tStandard Deviation of Fold Changes\n'
				all_pvals_data_zipped = zip(proteins, pvals, pvals2, folds, stdevs)
				all_pvals_data_zipped = list(all_pvals_data_zipped)
				all_pvals_data_zipped.sort(key=lambda x: x[1], reverse=False)
				with open(name_prefix+'all_pvals_'+str(tags[i])+'_vs_'+str(tags[j])+'.tsv','w') as handle:
					handle.write(header)
					for id, init_pval, adj_pval, fold_change, stdev in all_pvals_data_zipped:
						handle.write(str(id) + "\t"
									 + str(init_pval) + "\t"
									 + str(adj_pval) + "\t"
									 + str(fold_change) + "\t"
									 + str(stdev) + "\n")

				with open(name_prefix+'top20/'+ 'all_pvals_'+str(tags[i])+'_vs_'+str(tags[j])+'_top20.tsv','w') as handle:
					handle.write(header)
					for k, (id, init_pval, adj_pval, fold_change, stdev) in enumerate(all_pvals_data_zipped):
						if k < 20:
							handle.write(str(id) + "\t"
									 + str(init_pval) + "\t"
									 + str(adj_pval) + "\t"
									 + str(fold_change) + "\t"
									 + str(stdev) + "\n")
	return output_message

		
def cluster_proteins_combined(proteins, data, filename, name, prots, data_proteins):
	'''
	Clusters proteins (combined).

	Args:
		proteins: list of proteins
		data: list of input data, as a list of lists
		filename: output filename
		name: name of combined cluster image
		prots: ???
		data_proteins: list of lists

	Returns: nothing, only writes the clusters to a file and produces the plots of the combined clusters.
	'''
	data_for_clustering=list()
	for j in range(len(proteins)):
		data_for_clustering.append([])
		splitted_data=list()
		for i in range(timepoints):
			if i>0:
				temp_table=[]
				for k in range(len(data)):
					if k%timepoints==i:
						temp_table.append(data[k][j])
				data_for_clustering[j].append(sum(temp_table)/float(len(temp_table)))
		splitted_data.append(temp_table)

	for j in range(len(prots)):
		data_for_clustering.append([])	
		splitted_data=list()
		for i in range(timepoints):
			if i>0:
				temp_table=[]
				for k in range(len(data_proteins)):
					if k%timepoints==i:
						temp_table.append(data_proteins[k][j])
				data_for_clustering[j+len(proteins)].append(sum(temp_table)/float(len(temp_table)))
		splitted_data.append(temp_table)

	db = DBSCAN(eps=0.5, min_samples=2).fit(data_for_clustering)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print('Estimated number of clusters: %d' % n_clusters_)
	
	print("Silhouette Coefficient: %0.3f"
		  % metrics.silhouette_score(data_for_clustering, labels))

	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	print(unique_labels)
	colors = [plt.cm.Spectral(each)
			  for each in np.linspace(0, 1, len(unique_labels))]
	resultFile= open(filename,'w') 
	message=''
	for k, col in zip(unique_labels, colors):		
		class_member_mask = (labels == k)
		data_for_clustering_filtered=list()
		mpl_fig = plt.figure()
		ax = mpl_fig.add_subplot(111)
		protein_list=''
		for iter in range(len(data_for_clustering)):
			#print(labels)
			#print(iter)
			if labels[iter]==k:
				#data_for_clustering_filtered.append(copy.deepcopy(data_for_clustering_filtered))
				if (iter>=len(proteins)):
					protein_list=protein_list+'\t'+prots[iter-len(proteins)]
				else:
					protein_list=protein_list+'\t'+proteins[iter]
				ax.plot(data_for_clustering[iter])					
		ax.set_title('Cluster_'+str(k))
		ax.set_ylabel('Average Folded Concetrations')
		ax.set_xlabel('Time-points')
		#labels_n=list()
		labels_n = [item.get_text() for item in ax.get_xticklabels()]
		
		labels_n [1] = '0'
		labels_n [3] ='60min'
		labels_n [5] = '8h'
		labels_n [7] ='24h'
		ax.set_xticklabels(labels_n)
		#mpl_fig.Title(proteins[j])
		mpl_fig.savefig(name+'_Combined_Cluster_'+str(k)+'_figure.png')
		mpl_fig.clf()
		
		message+='Cluster_'+str(k)+':'+protein_list+'\n'
		#
	resultFile.write(message)
	resultFile.close()	
	
			
def create_combined_data(proteins, data, prots, data_proteins):
	'''
	Creates combined data.

	Args:
		proteins: list of proteins
		data: list of input data				
		prots: list of proteins ???
		data_proteins: data of proteins ???

	Returns: a list of two lists, variables and data_for_clustering

	Note: much of the code is repeating. Think about refactoring.
	'''
	data_for_clustering=list()#
	variables=list()
	#print(data)
	for j in range(len(proteins)):
		variables.append(proteins[j])
		data_for_clustering.append([])
		#data_for_clustering[j].append(proteins[j])
		splitted_data=list()
		for i in range(timepoints):
			if i>0:
				temp_table=[]
				for k in range(len(data)):
					if k%timepoints==i:
						data_for_clustering[j].append(data[k][j])
				
	for j in range(len(prots)):
		variables.append(prots[j])
		data_for_clustering.append([])
		#data_for_clustering[j].append(proteins[j])
		splitted_data=list()
		for i in range(timepoints):
			if i>0:
				temp_table=[]
				for k in range(len(data_proteins)):
					if k%timepoints==i:
						data_for_clustering[j+len(proteins)].append(data_proteins[k][j])		
	return [variables,data_for_clustering]
	
	
		
def create_combined_data_plasma(prots, data_proteins):
	'''
	Creates combined data plasma.

	Args:				
		prots: list of proteins ?
		data_proteins: list of lists, input data

	Returns: a list with data_for_clustering, variables

	Note: the code is the same with the code of the create_combined_data() function. Keep only one?
	'''
	data_for_clustering=list()
	variables=list()						
	for j in range(len(prots)):
		variables.append(prots[j])
		data_for_clustering.append([])		
		splitted_data=list()
		for i in range(timepoints):
			if i>0:
				temp_table=[]
				for k in range(len(data_proteins)):
					if k%timepoints==i:
						data_for_clustering[j].append(data_proteins[k][j])		
	return [variables,data_for_clustering]
	

def spearman_correlation(markers, data, filesuffix):
	'''
	Calculates spearman correlation coefficient.

	Args:
		markers: list with biomarkers
		data: input data, as a list of lists
		filesuffix: the suffix of the output file

	Returns: nothing, only writes the data in the spearman_table and spearman_network files.
	'''
	spearman_table= open(filesuffix+'spearman_table.txt','w')
	spearman_network=open(filesuffix+'spearman_network.txt','w')
	num_of_pairs=0
	for i in range(len(markers)):
		spearman_table.write('\t'+markers[i])
	spearman_table.write('\n')
	for i in range(len(markers)):
		spearman_table.write(markers[i])
		for j in range(len(markers)):
			if j>i:				
				[r,p]=st.spearmanr(data[i],data[j])
				spearman_table.write('\t'+str(r))
				if p<0.05 and r>0.5:
					spearman_network.write(markers[i]+'\t'+markers[j]+'\t'+str(r)+'\n')
			elif j==i:
				spearman_table.write('\t1')
			else:
				spearman_table.write('\t')
		spearman_table.write('\n')
					

def spearman_correlation_visualize(markers, data, output_folder):
	data = np.transpose(data)
	df = pd.DataFrame(data = data, columns = markers)	
	corr = df.corr()	
	fig, ax = plt.subplots(figsize=(10, 10))
	ax.matshow(corr)	
	#plt.xticks(range(len(corr.columns)), corr.columns)
	#plt.yticks(range(len(corr.columns)), corr.columns)	
	plt.savefig(output_folder + "pearson_matrix.png")

def filter_dataset(dataset_initial, proteins, percentage, output_message):
	'''
	Filters dataset.

	Args:
		dataset_initial: the initial dataset, a list of lists
		proteins: list of proteins
		percentage: percentage for filtering 
		output_message: resulting message

	Returns: a list with the new_data, new_proteins and the output_message.
	'''
	new_data=list()
	selected=0
	new_proteins=list()
	missing_proteins=0
	proteins_missing_values_percentage=0		
	for i in range(len(dataset_initial)):
		missing=0
		
		for j in range (len(dataset_initial[0])):
			if (dataset_initial[i][j]=='') or dataset_initial[i][j]==-1000:
				missing+=1
				proteins_missing_values_percentage+=1
					
		
		if ((missing/float(len(dataset_initial[0])))<=(percentage)):
			#print(i)
			new_data.append([])
			for k in range(len(dataset_initial[i])):
				new_data[selected].append(dataset_initial[i][k])
			selected+=1
			new_proteins.append(proteins[i])
		else:
			
			missing_proteins+=1
	print('Data were successfully filtered!')
	print('Total Number of Molecules='+str(len(dataset_initial)))
	print('Total Number of Molecules with missing values less than less than allowed threshold='+str(selected))
	print('Percentage of Missing Values in all molecules='+str(proteins_missing_values_percentage/float(len(dataset_initial)*len(dataset_initial[0]))))
	output_message+='Total Number of Molecules='+str(len(dataset_initial))+'\n'
	output_message+='Total Number of Molecules with missing values less than less than allowed threshold='+str(selected)+'\n'
	output_message+='Percentage of Missing Values in all molecules='+str(proteins_missing_values_percentage/float(len(dataset_initial)*len(dataset_initial[0])))+'\n'
	return [new_data,new_proteins,output_message]

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

	
def perform_missing_values_imputation(dataset_initial, missing_imputation_method, folder_name, output_message):
	'''
	Performs missing values imputation.

	Args:
		dataset_initial: the initial dataset, a list of lists
		missing_imputation_method (integer): if 1 choose average imputation, else choose KNN-imputation
		folder_name: output folder name
		output_message: output message which informs which imputation method has been used

	Returns: a list with final dataset (list of lists) and the output message.
	'''
	#missing values imputation
	averages=[0]*(len(dataset_initial))
	if missing_imputation_method==1:
		#average imputation here
		num_of_non_missing_values=[0]*(len(dataset_initial))
		
		for j in range(len(dataset_initial)):
			for i in range((len(dataset_initial[0]))):
				if dataset_initial[j][i]!=-1000 and dataset_initial[j][i]!='':
					#print(dataset_initial[j][i+1])
					averages[j]+=float(dataset_initial[j][i])
					num_of_non_missing_values[j]+=1					
			averages[j]=averages[j]/float(num_of_non_missing_values[j])

		write_one_dimensional_list_to_tab_delimited_file(averages, folder_name+'averages_for_missing_values_imputation.csv')
		output_message+='Average imputation method was used!\n'
		
		for i in range(len(dataset_initial)):
			for j in range((len(dataset_initial[0]))):
				if dataset_initial[i][j]==-1000:
					dataset_initial[i][j]=averages[i]

		#return dataset_initial # why not [dataset_initial, output_message] ???
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
	

def normalize_dataset(dataset_initial, folder_name, output_message, normalization_method):
	'''
	Performs normalization of the dataset.

	Args:
		dataset_initial: a list of lists, the initial dataset
		folder_name: the output folder name
		output_message: the message which informs which normalization method has been chosen
		normalization_method (integer): if 1 then arithmetic sample-wise normalization, else logarithmic normalization

	Returns: a list of the normalized data with the output message.
	'''		
	if normalization_method==1:
		#arithmetic sample-wise normalization
		maximums=[-1000.0]*(len(dataset_initial[0]))
		minimums=[1000.0]*(len(dataset_initial[0]))
		for i in range(len(dataset_initial)):
			for j in range((len(dataset_initial[0]))):
				if(dataset_initial[i][j]!="" and dataset_initial[i][j]!=-1000):
					if float(dataset_initial[i][j])>maximums[j]:
						maximums[j]=float(dataset_initial[i][j])
					if float(dataset_initial[i][j])<minimums[j]:
						minimums[j]=float(dataset_initial[i][j])
		max1=max(maximums)
		min1=min(minimums)
		print('Maximum Quantity Value:'+str(max1))
		print('Minimum Quantity Value:'+str(min1))
		for i in range(len(dataset_initial)):
			for j in range((len(dataset_initial[0]))):
				if (dataset_initial[i][j]!="" and dataset_initial[i][j]!=-1000):
					dataset_initial[i][j]=0+(1/(max1-min1))*(float(dataset_initial[i][j])-min1)		
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
	

def print_data(data, markers, tags, folder_name, filename, number_of_phenotypes):
	'''
	Prints data.

	Args:
		data: input data, list of lists
		markers: list of biomarkers
		tags: list of tags
		folder_name: output folder name
		filename: output filename
		number_of_phenotypes: number of phenotypes

	Returns: nothing, only writes the tags, the final data and the biomarkers to a file.
	'''
	file=open(folder_name+filename,'w')
	message=''
	for i in range(len(data[0])):
		message=message+'\t'+tags[i%number_of_phenotypes]
	message+='\n'
	for i in range(len(data)):
		message+=markers[i]
		for j in range(len(data[0])):
			message+='\t'+str(data[i][j])
		message+='\n'
	file.write(message)
	file.close()

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


def print_heatmap(proteins, data, labels, unique_labels, samples, scaling, filename):
	'''
	Prints heatmap.

	Args:
		proteins: list of proteins
		data: list of lists, input data
		labels: list of labels
		unique_labels: list of unique labeles
		samples: list of samples
		scaling: parameter for scaling
		filename: output filename

	Returns: prints a heatmap as a png.
	'''
	samples_colors=list()
	col_palet=list()
	col_palet.append("#0000FF")
	col_palet.append("#FF0000")
	col_palet.append("#00FF00")
	col_palet.append("#FFFF00")
	col_palet.append("#00FFFF")
	
	if len(unique_labels)<=5:
		max_num_of_colors=len(unique_labels)
	else:
		max_num_of_colors=5
	for i in range(len(data[0])):
		for k in range(max_num_of_colors):
			if labels[i]==unique_labels[k]:
				samples_colors.append(col_palet[k])
	#print('Heatmaps as PNG')
	logging.info('Heatmaps as PNG')
	r.png(filename, width = 10, height = 10,units = 'in',res = 1200)
	data_array=np.asarray(data)	
	#r.heatmap(data_array)
	#print(len(data_array))
	#print(len(np.asarray(samples_colors)))
	r.heatmap(data_array, cexRow=scaling, labCol=np.asarray(samples),labRow=np.asarray(proteins),ColSideColors = np.asarray(samples_colors), 
			col = ro.r.redgreen(75), show_heatmap_legend = True, show_annotation_legend = True)
	r("dev.off()")	


def print_significant_data_better(data, markers, samples, pvals, p_value_threshold, output_folder):
	"""
	Prints significant data to file, by filtering using the p_value_threshold.

	Args:
		 data (list): the list of lists with the data
		 markers (list): the list with all the markers names
		 samples (list): the list with the sample names
		 pvals (list): the list with the p-Value values
		 p_value_threshold (float): the p value threshold which is used for filtering
		 output_folder (string): the output folder name
	"""
	new_proteins = list()
	for j in range(len(markers)):
		if pvals[j] < float(p_value_threshold):
			new_proteins.append(markers[j])
	indexes = [markers.index(new_proteins[i]) for i in range(len(new_proteins))]
	new_data = [data[i] for i in indexes]
	print_data2(new_data, new_proteins, samples, output_folder, "significant_molecules_dataset.tsv")


def print_pvalues_tofile(markers, pvals, adj_pvals, output_folder, output_filename):
	"""
	Prints to file the pvalues data for each marker, along with the adjuste pvalue data, sorted per pvalue (ascending)

	Args:
		 markers (list): the list with the marker names
		 pvals (list): the list with the p-Value values
		 adj_pvals (list): the list with the adjusted p-Value values
		 output_folder (string): the output folder name
		 output_filename (string): the name of the output file

	Returns:
		prints data to filename named output_filename
	"""
	header = 'ID\tPvalue\tAdjusted Pvalues\n'
	all_pvals_data_zipped = zip(markers, pvals, adj_pvals)
	all_pvals_data_zipped = list(all_pvals_data_zipped)
	all_pvals_data_zipped.sort(key=lambda x: x[1], reverse=False)
	with open("{0}{1}".format(output_folder, output_filename), 'w') as handle:
		handle.write(header)
		for id, pval, adj_pval in all_pvals_data_zipped:
			handle.write(str(id) + "\t"
						 + str(pval) + "\t"
						 + str(adj_pval) + "\n")

def print_pvalues_tofileTop20(markers, pvals, adj_pvals, output_folder, output_filename):
	"""
	Prints to file the pvalues data for each marker, along with the adjuste pvalue data, sorted per pvalue (ascending)

	Args:
		 markers (list): the list with the marker names
		 pvals (list): the list with the p-Value values
		 adj_pvals (list): the list with the adjusted p-Value values
		 output_folder (string): the output folder name
		 output_filename (string): the name of the output file

	Returns:
		prints data to filename named output_filename
	"""
	header = 'ID\tPvalue\tAdjusted Pvalues\n'
	all_pvals_data_zipped = zip(markers, pvals, adj_pvals)
	all_pvals_data_zipped = list(all_pvals_data_zipped)
	all_pvals_data_zipped.sort(key=lambda x: x[1], reverse=False)
	with open("{0}{1}".format(output_folder, output_filename), 'w') as handle:
		handle.write(header)
		for i, (id, pval, adj_pval) in enumerate(all_pvals_data_zipped):
			if i < 20:
				handle.write(str(id) + "\t"
						 + str(pval) + "\t"
						 + str(adj_pval) + "\n")

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
				#new_samples.append(samples[i])		# file without samples in this version
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

def new_parse_data(data_filename, delimiter):
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

def get_pairs_from_list(alist:list)->list:
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

def run_statistical_data_analysis_multiple_conditions(data, phenotypes, markers, samples, parametric_flag,
													  p_value_threshold, paired_flag, logged_flag, folder_name):
	'''
	Runs the statistical data analysis script for multiple conditions.

	Args:
		data: the input data
		phenotypes: the labels
		markers (list): the marker names list
		samples: the samples names
		parametric_flag (int): if 0 then you don't know if parametric or not parametric analysis should be performed, if 1 parametric analysis will be performed and
						 if 2 non-parametric analysis will be performed.
		p_value_threshold: p-value threshold
		paired_flag: needed to perform pairwise analysis
		logged_flag: 0 if data not logged, 1 if logged
		folder_name: the output folder name

	Returns: [1, string1] for executing successfully and [0, string2] for failing to execute.
	'''
	
	output_message = ''
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
		
	labels = phenotypes

	tags = list(set(labels))

	# Transpose data for kruskal/anova, print_significant_boxplots, pairwise_analysis
	data_transp = np.transpose(data)
	plotly.tools.set_credentials_file(username='theofilk', api_key='FvCv1RgpeQC6ng3rxNfR')

	try:
		# If you don't know if parametric or not parametric analysis should be performed
		if parametric_flag == 0:
			data_per_phenotype = list()
			shapiro_pvals = list()
			flag = 0
			for k in range(len(tags)):
				data_per_phenotype.append([])
				for i in range(len(data)):
					for j in range(len(data[i])):
						if labels[j] == tags[k]:
							data_per_phenotype[k].append(data[i][j])
				
				[shapiro, shapiro_pvals_temp] = st.shapiro(data_per_phenotype[k])
				shapiro_pvals.append(shapiro_pvals_temp)
				if shapiro_pvals_temp < 0.05:
					flag = 1
			if flag == 1:
				[Hs, pvals] = kruskal_wallis_test(data_transp, labels)
			else:
				[Hs, pvals] = anova_test(data_transp, labels)
			if flag == 1:
				filename = 'kruskal_wallis_test.csv'
			else:
				filename = 'anova_test.csv'
			pvalues = smm.multipletests(pvals,method='fdr_bh')
			pvals2 = pvalues[1]
			print_pvalues_tofile(markers, pvals, pvals2, folder_name, filename)
			os.mkdir(folder_name + "top20/")
			print_pvalues_tofileTop20(markers, pvals, pvals2, folder_name + "top20/", filename + "_top20.csv")

			if flag == 1:
				output_message += 'Kruskal Wallis test was used for multple phenotypes comparison because at least for one of the categories data are not normally distributed.\n'
				logging.info('Kruskal Wallis test was used for multple phenotypes comparison because at least for one of the categories data are not normally distributed.')
				parametric_flag = 2
			else:
				output_message += 'Anova test was used for multple phenotypes comparison because at least for one of the categories data are not normally distributed.\n'
				logging.info('Anova test was used for multple phenotypes comparison because at least for one of the categories data are not normally distributed.')
				parametric_flag = 1
		# If  you want to perform parametric analysis
		elif parametric_flag == 1:
			[Hs, pvals] = anova_test(data_transp, labels)
			pvalues = smm.multipletests(pvals, method='fdr_bh')
			pvals2 = pvalues[1]
			filename = 'anova_test.csv'
			print_pvalues_tofile(markers, pvals, pvals2, folder_name, filename)
			os.mkdir(folder_name+ "top20/")
			print_pvalues_tofileTop20(markers, pvals, pvals2, folder_name + "top20/", filename + "_top20.csv")

			output_message += 'Anova test was used for multiple phenotypes comparison.\n'
			logging.info('Anova test was used for multiple phenotypes comparison.')
		# If you want to perform non-parametric analysis
		else:
			[Hs, pvals] = kruskal_wallis_test(data_transp, labels)
			pvalues = smm.multipletests(pvals, method='fdr_bh')
			pvals2 = pvalues[1]
			filename = 'kruskal_wallis_test.csv'
			print_pvalues_tofile(markers, pvals, pvals2, folder_name, filename)
			os.mkdir(folder_name + "top20/")
			print_pvalues_tofileTop20(markers, pvals, pvals2, folder_name + "top20/", filename + "_top20.csv")

			output_message += 'Kruskal Wallis test was used for multple phenotypes comparison.\n'
			logging.info('Kruskal Wallis test was used for multple phenotypes comparison.')
		
		#Step 8 Keep Significant Data Only
		print_significant_boxplots(markers, data_transp, labels, folder_name+'boxplots/', 0, pvals,tags,p_value_threshold)
		try:
			print_heatmap(markers,data,labels,tags, samples,0.4, folder_name+'heatmap_all.png')
		except Exception as e:
			logging.exception("Exception during printing heatmap_all.png")
		
		if parametric_flag in {0,1,2}:
			protein = markers
			proteins2 = markers
			dataset_imputed = data
			filtered_data = list()
			position = 0
			diff_proteins_corrected = list()
			for i in range(len(protein)):
				prot = protein[i]
				if pvals[i] < p_value_threshold:
					diff_proteins_corrected.append(prot)
					ind = proteins2.index(prot)
					filtered_data.append([])
						
					for j in range(len(dataset_imputed[ind])):
						filtered_data[position].append(dataset_imputed[ind][j])
					position += 1
			if position > 1:	# heatmap must have at least 2 rows and 2 columns
				try:
					print_heatmap(diff_proteins_corrected,filtered_data,labels,tags, samples,0.4, folder_name + "heatmap_not_corrected.png")
				except Exception as e:
					logging.exception("Exception during printing heatmap_not_corrected.png")
				# printing significant boxplots data
			if position >= 1:
				print_data2(filtered_data, diff_proteins_corrected, samples, folder_name, "significant_molecules_dataset.tsv")

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
					print_heatmap(diff_proteins_corrected,filtered_data,labels,tags, samples,0.4, folder_name + "heatmap_corrected.png")
				except:
					logging.exception("Exception during printing heatmap corrected.png")

		#Step 11 Perform Pairwise Statistical Analysis
		try:
			output_message = perform_pairwise_analysis(markers, data_transp, folder_name, labels, tags, parametric_flag,
													   p_value_threshold, output_message, paired_flag)
		except Exception as e:
			logging.exception("Statistical analysis with multiple conditions raised exception during pairwise analysis.")
			return [0, "Statistical analysis with multiple conditions raised exception pairwise analysis: {}.".format(str(e))]
		out_file = open(folder_name + 'info.txt','w')
		out_file.write(output_message)
		out_file.close()
	except Exception as e:
		logging.exception("Statistical analysis with multiple conditions raised the exception during testing.")
		return [0, "Statistical analysis with multiple conditions raised the exception during testing: {}".format(str(e))]

	try:
		# Printing the MQ files for the full dataset
		os.mkdir(folder_name + "MQ_files")
		create_MQ_files(data, markers, labels, "MQ_", "{0}MQ_files/".format(folder_name))

		# First parsing the significant  molecules dataset
		filename = folder_name + "significant_molecules_dataset.tsv"
		if os.path.isfile(filename):
			os.mkdir(folder_name + "MQ_significant_files")
			sign_markers, sign_dataset, samples = parse_data_all_headers(filename)
			# Printing the MQ files for the significant dataset
			create_MQ_files(sign_dataset, sign_markers, labels, "MQ_significant_",
							'{0}MQ_significant_files/'.format(folder_name))
	except Exception as e:
		logging.exception("Statistical analysis raised an exception during creation of MQ files.")
		return [0, "Statistical analysis raised an exception during creation of MQ files: {}. ".format(str(e))]

	try:
		if os.path.isfile(filename):
			# Creating differential expression files, for all pairs of MQ files
			os.mkdir(folder_name + "diff_expression_files")
			input_files = os.listdir(folder_name + "MQ_significant_files/")
			pairs_of_names = get_pairs_from_list(input_files)
			for pair in pairs_of_names:
				proteins1, dataset1 = new_parse_data(folder_name + "MQ_significant_files/" + pair[0], "\t")
				proteins2, dataset2 = new_parse_data(folder_name + "MQ_significant_files/" + pair[1], "\t")
				if parametric_flag in {1,2}:
					create_differential_expression_file(dataset1, dataset2, proteins1, logged_flag, p_value_threshold, parametric_flag,
													folder_name + "diff_expression_files/",
													"diff_express_file_{0}_VS_{1}.tsv".format(pair[0][:-4],pair[1][:-4]))
				else:
					zero_parametric_flag = 1+flag # 1 for parametric, 2 for non parametric
					create_differential_expression_file(dataset1, dataset2, proteins1, logged_flag, p_value_threshold,
														zero_parametric_flag,
														folder_name + "diff_expression_files/",
														"diff_express_file_{0}_VS_{1}.tsv".format(pair[0][:-4], pair[1][:-4]))

	except Exception as e:
		logging.exception("Statistical analysis with multiple conditions raised an exception during creation of"
						  "differential expression files.")
		return [0, "Statistical analysis with multiple conditions raised an exception during creation of "
				   "differential expression files: {}".format(str(e))]

	return [1, "Job completed successfully."]
	
if __name__ == "__main__":
	print(0)
	#Example run: python statistical_data_analysis_multiple_conditions_v3_0.py example_dataset.txt example_labels.txt example_samples.txt 0 2 2 0.3 0.05 0 example_results/
	#result = run_statistical_data_analysis_multiple_conditions("example_dataset.txt", "example_labels.txt", "example_samples.txt", 1, 1, 1, 0.3, 0.05, 0, "example_results5/")
	#print(result)

	
	
