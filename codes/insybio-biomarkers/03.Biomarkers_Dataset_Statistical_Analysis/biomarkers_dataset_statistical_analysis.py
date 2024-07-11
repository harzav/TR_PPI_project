"""
Script to perform the statistical analysis of the biomarkers dataset.

Example run:
	python3 biomarkers_dataset_statistical_analysis.py Input/Single/example_dataset_preprocessed.txt
	Input/Single/example_labels.txt "ACTB,VIME,APOE,TLN1,CO6A3" Output/ 19 1 1 0 0.05 0
	python3 biomarkers_dataset_statistical_analysis.py Input/Single/output_dataset_single_with_geo4.txt
	Input/Single/example_labels.txt "ACTB,VIME,APOE,TLN1,CO6A3" Output/ 19 1 1 0 0.05 0
	python3 biomarkers_dataset_statistical_analysis.py input/dataset_without_samples.txt labels.txt
	"ACTB,VIME,APOE,TLN1,CO6A3" Output/ 7 1 0 0 0.05 0
	python3 biomarkers_dataset_statistical_analysis.py Input/Single/output_dataset_single_with_geo5_nosamples2.txt
	"ACTB,VIME,APOE,TLN1,CO6A3" Output/ 7 1 0 0 0.05 0
	python3 biomarkers_dataset_statistical_analysis.py Input/Multiple/output_dataset4b.txt
	Input/Multiple/example_labels.txt "" Output/ 19 1 1 0 0.05 2
"""

import configparser
import datetime
import logging
from statistical_analysis_v4_3 import run_statistical_analysis
from statistical_data_analysis_multiple_conditions_v3_1 import run_statistical_data_analysis_multiple_conditions
import os
import sys
import csv
import numpy as np
import subprocess
import json


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
		filename="{}biomarkers_reports_stat_analysis_{}.log".format(config["logs"]["logpath"], todaystr),
		level=logging.DEBUG, format='%(asctime)s\t %(levelname)s\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


def write_one_dimensional_list_to_tab_delimited_file(data, filename):
	"""
	Writes one dimensional list to tab delimited file.

	Args:
		data: input data
		filename: output filename

	Returns: doesn't return anything, only writes data to file.
	"""
	with open(filename,'w') as file_id:
		for i in range(len(data)):
			file_id.write(str(data[i]))
			if i < len(data) - 1:
				file_id.write('\t')


def file_get_contents(filename):
	"""
	Gives the contents of a file using f.read() method.

	Args:
		filename: the filename
	"""
	with open(filename) as f:
		return f.read()


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
						if line[j] != '':
							data[num_of_lines-1].append(float(line[j]))
						else:
							data[num_of_lines-1].append('')
			num_of_lines += 1
	# print('Data were successfully parsed!')
	logging.debug('Data were successfully parsed!')
	return [proteins, data, samples]


def parse_data2(data_filename):
	"""
	Parses data.

	Args:
		data_filename: input data filename

	Returns: a list with the proteins and the data.
	"""
	num_of_lines = 0
	proteins = list()
	data = list()
	with open(data_filename) as data_fname:
		for line in csv.reader(data_fname, delimiter="\t"):
			if num_of_lines == 0:
				for j in range(len(line)):
					proteins.append(line[j].strip().replace(" ", "").replace('/', ''))
				else:

					data.append([])
					for j in range(len(line)):
						if line[j] != '' and line[j] != -1000:
							data[num_of_lines-1].append(float(line[j]))
						elif line[j] != '':
							data[num_of_lines-1].append(-1000)
						else:
							data[num_of_lines-1].append('')
				num_of_lines += 1
	# print('Data were successfully parsed!')
	logging.debug('Data were successfully parsed!')
	return [proteins, data]


def parse_labels_new(labels_filename, delimiter):
	"""
	Parses labels.

	Args:
		labels_filename (string): the labels filename
		delimiter (string): the kind of delimiter with values "," or "\t"

	Returns: a list with the labels.
	"""
	with open(labels_filename, 'r') as labels_fln:
		labels = list()
		for line in labels_fln:
			words = line.split(delimiter)
			# try:
			#    words = [int(x) for x in words]
			# except:
			#    words = [x.strip() for x in words]
			words = [x.strip() for x in words]
			labels.append(words)
	logging.info("Labels successfully parsed!")
	return labels


def parse_labels_generic(labels_dataset, delimiter):
	"""
	Parses labels file.

	Args:
		labels_dataset (string): the labels filename
		delimiter (string): "\t" for TSV or "," for CSV

	Returns: a list with the labels as strings
	"""
	labels = list()
	with open(labels_dataset, 'r') as handle:
		for line in csv.reader(handle, delimiter = delimiter):
			for i in range(len(line)):
				labels.append(line[i].strip())	
	return labels


def parse_selected_features(features_filename):
	"""
	Parses the selected features filename.

	Args:
		features_filename: the selected features filename, one line tab seperated values

	Returns:
		features (list): the list of the selected features
	"""
	features = list()
	num_of_lines = 0
	try:
		with open(features_filename) as features_fname:
			for line in csv.reader(features_fname, delimiter="\t"):
				for i in range(len(line)):
					features.append(line[i].strip()) # returns a list of one string eg. ['1 2 3']
		features = list(map(int, features[0].split())) # returns a list of ints eg. [1,2,3]
		# print('Features were successfully parsed!')
		logging.debug('Features were successfully parsed!')
		return features
	except Exception:
		# print('Empty selected features file provided!')
		logging.debug('Empty selected features file provided!')


def parse_samples2(samples_filename):
	"""
	Parses the samples.

	Args:
		samples_filename:	the samples filename

	Returns: a list with the samples.
	"""
	samples = list()
	with open(samples_filename) as samples_fname:
		for line in samples_fname:
			words = line.split('\t')
			for sample in words:
				samples.append(sample.strip())
	return samples


def parse_selected_features_string(astring):
	"""
	Parses a string and strips it from commas or newline characters.

	Args:
		astring: the input string with comma separated or newline separated values

	Returns:
		A list with the substrings of the original string.
	"""

	if "," in astring:
		return astring.split(",")
	elif "\\n" in astring:
		return astring.split("\\n")
	elif "\r\n" in astring:
		return astring.split("\r\n")
	elif "\n" in astring:
		return astring.split("\n")
	else:
		# raise ValueError("The string doesn't contain comma separated values or newline separated values !")
		return astring


def new_parse_data(data_filename,delimiter):
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
				if j>0:
					if line[j] != '':
						data[num_of_lines].append(float(line[j]))
					else:
						data[num_of_lines].append('')
			num_of_lines += 1
	# print('Data were successfully parsed!')
	logging.debug('Data were successfully parsed!')
	return [proteins, data]


def parse_data_with_only_samples(data_filename,delimiter):
	"""
	Parses data.

	Args:
		data_filename (string): dataset filename with only data and samples
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
						data[num_of_lines-1].append(float(line[j]))
					except Exception:
						data[num_of_lines-1].append('')
			num_of_lines += 1
	# print('Data were successfully parsed!')
	return [data, samples]


def parse_only_dataset(dataset_filename, delimiter):
	"""
	Parses a dataset which has no headers at all.

	Args:
		dataset_filename (string): the dataset filename
		delimiter (string): the kind of delimiter with values "," or "\t"		

	Returns:
		data (list): a list of lists with the data

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
	logging.info('Data were successfully parsed!')
	return data


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


def find_delimiter_labels(dataset_filename):
	"""
	Figures out which delimiter is being used in given labels dataset.

	Args:
		dataset_filename (string): the dataset filename

	Returns:
		(string): "," if CSV content, "\t" if TSV content.
	"""
	with open(dataset_filename, 'r') as handle:
		head = next(handle)
	if "\t" in head:
		return "\t"
	elif "," in head:
		return ","
	elif "," and "\t" in head:  # The case where the comma is the decimal separator (greek system)
		return "\t"


def create_samples_tags(input_dataset):
	"""
	Creates a list with the sample's tags, given the number of markers n.

	Args:
		input_dataset: input dataset

	Returns:
		alist: a list with the sample's tags
	"""
	n = len(input_dataset[0])
	alist = ["sample_" + str(i) for i in range(n)]
	return alist


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


def replace_strange_chars_from_markers(list_of_markers):
	"""
	Replaces the character "/" found in any markers names with "_".

	Args:
		list_of_markers (list): a list of strings

	Returns:
		(list): a one dimensional list with the markers names
	"""
	return [x.replace("/", "_") for x in list_of_markers]


def recognise_file_structure(input_dataset, labels_filename, delim):
	"""
	Handler. Recognises file structure and transpose the file if needed.

	:param input_dataset: input dataset
	:param labels_filename: labels file
	:param delim: delimiter of the file
	:return: input_dataset/ None
	"""
	try:
		if is_correct(input_dataset, labels_filename, delim):
			# if structure is correct we want to transpose it because the code deals with transpose files
			input_dataset = transpose_file(input_dataset, delim)
			return input_dataset
		elif is_transpose(input_dataset, labels_filename, delim):
			return input_dataset
		else:
			print("Error with file structure")
			return None
	except FileNotFoundError as e:
		return [None, "Code raised {}. Make sure you give the correct path".format(str(e))]


def transpose_file(input_dataset, delim, ret="False"):
	"""
	Creates transposed file
	:param input_dataset: the dataset to be transposed
	:param path: path to file
	:param delim: delimiter of file
	:return: None
	"""
	csv = pd.read_csv(input_dataset, delimiter=delim)
	df_csv = pd.DataFrame(data=csv)
	transposed_csv = df_csv.T
	transposed_csv.to_csv(input_dataset[:-4] + "_transpose.txt", sep='\t')
	print("transposed file created ")
	if ret == "True":
		return transposed_csv


def is_transpose(input_dataset, labels_filename, delim):
	"""
	Recognises if given file is transpose or not
	Transpose file correct dimensions: #labels == length of first column of data (samples)
	:param input_dataset:
	:param labels_filename:
	:return: True/False
	"""
	labels = pd.read_csv(labels_filename, sep=delim)
	data = pd.read_csv(input_dataset, sep=delim)
	if (data.shape[0] == labels.shape[1]) or (data.shape[0] == (labels.shape[1] - 1)):
		print("data are transpose with correct dimensions")
		return True
	else:
		print("data are not transpose")
		return False


def is_correct(input_dataset, labels_filename, delim):
	"""
	Recognises if file given has correct dimensions
	Correct dimensions: #labels == #samples == length of first row of data (samples)
	:param input_dataset:
	:param labels_filename:
	:return: True/False
	"""
	labels = pd.read_csv(labels_filename, sep=delim)
	data = pd.read_csv(input_dataset, sep=delim)
	if (data.shape[1] == labels.shape[1]) or (data.shape[1] == (labels.shape[1] - 1)):
		print("correct sizes")
		return True
	else:
		print("incorrect sizes")
		return False


def meta_statistical_analysis(
		input_dataset, labels_filename, selected_comorbidities_string, output_folder_name, filetype,
		has_features_header, has_samples_header, paired_flag=0, logged_flag=0, pvalue_threshold=0.05, parametric_flag=0,
		volcano_width=8, volcano_height=8, volcano_titles=8, volcano_axis_labels=8, volcano_labels=1,
		volcano_axis_relevance=32.0, volcano_criteria=3, abs_log_fold_changes_threshold=0.5, volcano_labeled=2,
		heatmap_width=15, heatmap_height=15, features_hier="hierarchical", features_metric="euclidean",
		features_linkage="complete", samples_hier="hierarchical", samples_metric="euclidean", samples_linkage="single",
		heatmap_zscore_bar=1, beanplot_width=12, beanplot_height=12, beanplot_axis=1.3, beanplot_xaxis=1.3,
		beanplot_yaxis=1.3, beanplot_titles=1.3, beanplot_axis_titles=1.2, user='unknown', jobid=0, pid=0,
		image_flag=True):
	"""
	Selects which script will do the statistical analysis.
	:param input_dataset: the initial dataset
	:param labels_filename: the labels filename
	:param selected_comorbidities_string: the string with the names of the selected genes that will be used to perform
	corrections on the statistical analysis, separated with commas or "\n"
	:param output_folder_name: output folder name
	:param filetype: if it is 7 then it doesn't have a samples header
	:param has_features_header: if it is 1 it has features header, 0 if it doesn't have
	:param has_samples_header: if it is 1 it has features header, 0 if it doesn't have
	:param paired_flag: 1 for paired analysis 0 for non paired analysis (needed for wilcoxon rank sum)
	:param logged_flag: 0 if data not logged, 1 if logged
	:param pvalue_threshold: p value threshold, used for filtering
	:param parametric_flag: 0 if you don't know if you want parametric or non-parametric testing to be applied,
	1  if you wish parametric testing to be applied, 2 if you wish non-parametric testing to be applied
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
	:param features_hier: An option between "hierarchical" and "none"
	:param features_metric: An option between "euclidean","manhattan","maximum"
	:param features_linkage: An option between "average","single","complete","centroid","ward.D"
	:param samples_hier: An option between "hierarchical" and "none"
	:param samples_metric: An option between "euclidean","manhattan","maximum"
	:param samples_linkage: An option between "average","single","complete","centroid","ward.D"
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
	heatmap_clust_features = [features_hier, features_metric, features_linkage]
	heatmap_clust_samples = [samples_hier, samples_metric, samples_linkage]

	try:
		# Parsing
		if not os.path.exists(output_folder_name):
			os.makedirs(output_folder_name)

		# Figure out the delimiter of the dataset
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

		# Replace "/" with "_" in markers names
		markers = replace_strange_chars_from_markers(markers)

		delim_labels = find_delimiter_labels(labels_filename)
		labels = parse_labels_new(labels_filename, delim_labels)  # labels are being parsed as a list of list(s)
		# logging.debug(labels)
		# selected_features = parse_selected_features(selected_features_filename)
		selected_comorbidities = parse_selected_features_string(selected_comorbidities_string)
		# print(selected_features)
		logging.debug(selected_comorbidities)
		if '' in selected_comorbidities and not isinstance(selected_comorbidities, str):
			selected_comorbidities.pop()
	except Exception as e:
		logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during"
						  " parsing.".format(pid, jobid, user), 0)
		return [0, "Statistical analysis raised the exception during parsing: {}".format(str(e))]
	isdifferent, msg = checklabel_and_data_length(data, labels)
	if isdifferent:
		return [0, "Statistical analysis error: Labels and Data length is different, pls check if they are "
				   "matching. {}".format(msg), 0]
	try:
		# Check if labels have more than two columns
		if len(labels) > 1:
			# print("More than one column of labels found!")
			logging.debug("More than one column of labels found!")
			multi_label_flag_list = [True if len(list(set(l))) > 2 else False for l in labels]

		else:
			logging.debug("Single column of labels found.")
			multi_label_flag = True if len(list(set(labels[0]))) > 2 else False

	except Exception as e:
		logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during label"
						  " checking.".format(pid, jobid, user))
		return [0, "Statistical analysis raised the exception during label checking: {}".format(str(e)), 0]

	try:
		# Creating the list of indexes of the selected features
		if selected_comorbidities:
			selected_comorbidities_nums = list()
			for comorbidity in selected_comorbidities:
				try:
					comorbidity_index = markers.index(comorbidity)
					selected_comorbidities_nums.append(comorbidity_index)
				except Exception as e:
					return [0, "The biomarker(s) provided are not in the list of the biomarkers of the input file !"
							   " Try again ! {}".format(e), 0]
			# print(selected_features_nums)
			logging.debug(selected_comorbidities_nums)

			commorbidities = np.transpose([data[i] for i in range(len(data)) if i in selected_comorbidities_nums])
			commorbidities_filename = open(output_folder_name + 'commorbidities.txt', 'w')
			np.savetxt(commorbidities_filename, commorbidities, delimiter='\t', fmt='%.2f')
			commorbidities_types = [1 if (isinstance(x, float) or isinstance(x, int)) else 0 for x in commorbidities[0]]

			write_one_dimensional_list_to_tab_delimited_file(commorbidities_types,
															 output_folder_name + 'commorbidities_types.txt')

		elif selected_comorbidities == '':
			# raise ValueError('Please provide commorbidities for single statistical analysis!')
			shell_command1 = "touch {}commorbidities.txt && touch {}commorbidities_types.txt".format(output_folder_name,
																									 output_folder_name)
			f = open("{}{}".format(output_folder_name, "stdout.txt"), "w")
			err = open("{}{}.err".format(output_folder_name, "commorbidities"), "w")
			try:
				retcode = subprocess.call(shell_command1,stdout=f, stderr=err, shell=True)
				if retcode == 0:
					print([1, "Commorbidity files created successfully."])
				else:
					errStr = file_get_contents("{}{}.err".format(output_folder_name, "commorbidities"))
					return [0, json.dumps(["Error in creating the commorbidity files: {}".format(errStr)]), 0]
			except OSError as e:
				logging.exception("Creation of commorbidities failed!")
				print("Execution failed:", e, file=sys.stderr)
				return [0, json.dumps(["Error in creating the commorbidity files.", e.errno, e.strerror]), 0]

	except Exception as e:
		logging.exception("PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during creation of "
						  "commorbidities.".format(pid, jobid, user))
		return [0, "Statistical analysis raised the exception during creation of commorbidities: {}".format(str(e)), 0]

	try:
		# For every phenotypic column(label column) if it takes more than two values then run multiple values script.
		# Else run single variables script.
		if len(labels) > 1:
			logging.debug("Labels: {}".format(len(labels)))
			labels_list = labels
			for i, labels in enumerate(labels_list):
				output_folder_name2 = output_folder_name + "Output_" + str(i) + "/"
				if not os.path.exists(output_folder_name2):
					os.makedirs(output_folder_name2)
				logging.debug("Started run number: {}".format(i))
				if multi_label_flag_list[i]:
					logging.debug("multiple conditions ")
					result = run_statistical_data_analysis_multiple_conditions(
						data, labels, markers, samples, output_folder_name + 'commorbidities.txt',
						output_folder_name + 'commorbidities_types.txt', parametric_flag, pvalue_threshold, paired_flag,
						logged_flag, output_folder_name2, volcano_width, volcano_height, volcano_titles,
						volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
						abs_log_fold_changes_threshold, volcano_labeled, heatmap_width, heatmap_height,
						heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar, beanplot_width,
						beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles,
						beanplot_axis_titles, pid, jobid, user, image_flag=image_flag)
				else:
					logging.debug("two conditions ")
					result = run_statistical_analysis(
						data, labels, markers, samples, output_folder_name + 'commorbidities.txt',
						output_folder_name + 'commorbidities_types.txt', parametric_flag, pvalue_threshold, paired_flag,
						logged_flag, output_folder_name2, volcano_width, volcano_height, volcano_titles,
						volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
						abs_log_fold_changes_threshold, volcano_labeled, heatmap_width, heatmap_height,
						heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar, beanplot_width,
						beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles, beanplot_axis_titles, pid,
						jobid, user, image_flag=image_flag)
				# print("Finished run number: ",i)
				logging.debug("Finished run number: {}".format(i))

			return result + [len(labels_list)]
		else:
			labels = labels[0]
			if multi_label_flag:
				logging.debug("multiple conditions ")
				result = run_statistical_data_analysis_multiple_conditions(
					data, labels, markers, samples, output_folder_name + 'commorbidities.txt',
					output_folder_name + 'commorbidities_types.txt', parametric_flag, pvalue_threshold, paired_flag,
					logged_flag, output_folder_name, volcano_width, volcano_height, volcano_titles,
					volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
					abs_log_fold_changes_threshold, volcano_labeled, heatmap_width, heatmap_height,
					heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar, beanplot_width,
					beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles,
					beanplot_axis_titles, pid, jobid, user, image_flag=image_flag)

			else:
				logging.debug("two conditions")
				result = run_statistical_analysis(
					data, labels, markers, samples, output_folder_name + 'commorbidities.txt',
					output_folder_name + 'commorbidities_types.txt', parametric_flag, pvalue_threshold, paired_flag,
					logged_flag, output_folder_name, volcano_width, volcano_height, volcano_titles,
					volcano_axis_labels, volcano_labels, volcano_axis_relevance, volcano_criteria,
					abs_log_fold_changes_threshold, volcano_labeled, heatmap_width, heatmap_height,
					heatmap_clust_features, heatmap_clust_samples, heatmap_zscore_bar, beanplot_width,
					beanplot_height, beanplot_axis, beanplot_xaxis, beanplot_yaxis, beanplot_titles,
					beanplot_axis_titles, pid, jobid, user, image_flag=image_flag)
			return result + [0]
	except Exception as e:
		logging.exception(
			"PID:{}\tJOB:{}\tUSER:{}\tStatistical analysis raised the exception during statistical "
			"analysis.".format(pid, jobid, user))
		return [0, "Statistical analysis raised the exception during statistical analysis: {}".format(str(e)), 0]



def checklabel_and_data_length(data, labels):
	"""
	Check if labels and dataset match
	:param data:
	:param labels:
	:return:
	"""
	if len(data[0]) == len(labels[0]):
		return False, 'Same length'
	elif len(data[0]) > len(labels[0]):
		return True, 'More Samples/Columns in dataset than Labels'
	elif len(data[0]) < len(labels[0]):
		return True, 'More Labels than Samples/Columns in dataset'
	else:
		return True, ''


if __name__ == "__main__":
	biomarkers_dataset = sys.argv[1]
	labels_filename = sys.argv[2]
	selected_comorbidities_string = sys.argv[3]
	output_folder_name = sys.argv[4]
	filetype = int(sys.argv[5])
	has_features_header = int(sys.argv[6])
	has_samples_header = int(sys.argv[7])
	paired_flag = int(sys.argv[8])
	logged_flag = int(sys.argv[9])
	pvalue_threshold = float(sys.argv[10])
	# parametric_flag = int(sys.argv[11])
	parametric_flag = (sys.argv[11])

	result = meta_statistical_analysis(
		biomarkers_dataset, labels_filename, selected_comorbidities_string, output_folder_name, filetype,
		has_features_header, has_samples_header, paired_flag, logged_flag, pvalue_threshold, parametric_flag)
	print(result)
