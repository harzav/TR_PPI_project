# TR-PPI project

This is the directory containing the necessary files and codes for running the classification and regression models of the TR-PPI project. 
This is a project aiming at PPI discovery through an Evolutionary Optimization Algorithm and quantification of the strength of PPIs through regression models. The regression models are based on an endpoint created by experimental ΔG affinity scores mined from PDBbind v2020118, SKEMPI 2.0119, MPAD120 and Binding Affinity Benchmark version 2 (BABv2). 
These models are to be used on the TR interactome, with the purpose of finding potential novel TR interactions and filtering the most probable ones.

The Flowchart of the proposed method and the Tutorial for the feature calculation and model creation are presented below.

<h2>A. Flowchart of the Proposed Method</h2>

![Picture1](https://github.com/user-attachments/assets/6c1a9181-17b8-40c6-ac23-96e9cdd750c8)

## <h2>B. Table of Contents- Tutorial for model creation </h2>

- [1. Creation of the training datasets- feature calculation](#1-creation-of-the-training-datasets--feature-calculation)
- [2. Model Creation and Testing](#2-model-creation-and-testing)
- [3. BML Analysis](#3-bml-analysis)

## 1. Creation of the training datasets - feature calculation

The code for calculating the features for all Datasets in the project (affinity test and training datasets, PPI test and training datasets and taste receptor dataset) can be found in the directory <code>codes</code>.
‘feature_calculation_code.py’

Receives as input: A csv dataset with 6 columns that must be named ‘uidA’, ‘uidB’, protein_accession_A’, ‘protein_accession_B’, 'seq_A', 'seq_B'. It can have more columns but these 6 must be present.

![image](https://github.com/user-attachments/assets/90b12bbb-ebf7-43c7-b0c7-cb9e387a3332)

Output: A csv dataset with 61 additional features named:  'BP_similarity', 'MF_similarity', 'CC_similarity', 'Exists in MINT?', 'Exists in DIP?', 'Exists in APID?','Exists in BIOGRID?', 'Sequence_similarity', 'pfam_interaction','MW dif', 'Aromaticity dif', 'Instability dif', 'helix_fraction_dif', 'turn_fraction_dif', 'sheet_fraction_dif', 'cys_reduced_dif', 'gravy_dif', 'ph7_charge_dif', 'A %', 'L %', 'F %', 'I %', 'M %', 'V %', 'S %', 'P %', 'T %', 'Y %', 'H %', 'Q %', 'N %', 'K %', 'D %', 'E %', 'C %', 'W %', 'R %', 'G %',   'GSE227375_spearman', 'GSE228702_spearman', ‘0, 1, 2...14’, ‘Homologous in Mouse/ Drosophila/Yeast/Ecoli ’,  ‘Subcellular Co-localization?’

Output examples from the different datasets created for this project can be found in the <code>example_datasets</code> directory (AFFINITY_DATASET, PPI_TESTING_DATASET,PPI_TRAINING_DATASET)

The 'AFFINITY_DATASET' folder contains also the <code>'exp_affinities_ds.csv'</code>  file, which contains the mined experimental affinities for specific PPI pairs.

<code>Datasets</code>: The directory containing all the datasets needed for the calculation of the features. Note that the Download paths for these datasets must be passed manually to the feature calculation code python file, in order for it to run.

## 2. Model Creation and Testing

**The scripts for training and testing the models can be found in the directory <code>codes</code>.**

For training:

<code>insybio-biomarkers</code> > <code> 04.Training_Multibiomarker_Predictive_Analytics_Model </code> > <code> biomarker_discovery_script_selection_backend.py </code>

For testing the classification models:

<code>insybio-biomarkers</code> > <code> 05.Testing_Multibiomarker_Predictive_Analytics_Model </code> > <code> testing_multibiomarkers_discovery_script_selection_backend.py </code>

For testing the regression models use the python script:
<code> regression_test.py </code> .
This includes extensive commentary on the input files and datasets that have to be used.

**The Unix commands for training the models are:**

For the Binary Classifier:
```
sudo python3 biomarker_discovery_script_selection_backend.py ~/Downloads/ppi_training_dataset.txt ~/Downloads/ppi_training_labelss.txt "1,10,10,1,1,1,1,1" "" 2 0 0 20 50 0.01 0 0.9 10 8 1 1 ~/Downloads/Output_folder_classifier/
```

For the Regressor:
```
sudo python3 biomarker_discovery_script_selection_backend.py ~/Downloads/dg_dataset_t.csv ~/Downloads/dg_labels.txt "1,10,1,1,1,1,1,1" "" 1 0 0 50 1000 0.01 0 0.9 10 8 1 1 ~/Downloads/Output_folder_regressor/
```

**Training Algorithm Parameters:**
<table>
  <tr>
   <th>Position</th>
   <th>Parameter</th>
   <th>Description</th>
   <th>Default value</th>
   
  </tr>
  <tr>
   <th>0</th>
   <td>Dataset Filename</td>
   <td>Input dataset that has features as rows and samples as columns</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>1</th>
   <td>Labels Filename</td>
   <td>Tab delimited txt file with label values</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>2</th>
   <td>Goal Significances String</td>
   <td>String with the significances of each goal-metric,containing comma separated integers</td>
   <td>"1,10,10,1,1,1,1,1"</td>
  </tr>
  <tr>
   <th>3</th>
   <td>Select Comorbidities String</td>
   <td>String with the selected comorbidities names, separated by commas or newline characters</td>
   <td>NaN</td>
  </tr>
  <tr>
   <th>4</th>
   <td>Selection Flag</td>
   <td>Integer indicating the type of model analysis.</td>
   <td>0 for multiclass classification, 1 for regression, 2 for two-class classification</td>
  </tr>
  <tr>
   <th>5</th>
   <td>Split Dataset Flag</td>
   <td>Integer indicating the dataset split. 1 if there will be train- test split, 0 if not</td>
   <td>0</td>
  </tr>
  <tr>
   <th>6</th>
   <td>Filtering Percentage</td>
   <td>The percentage of the total dataset that will be contained in the test set (if Split Dataset Flag= 1)</td>
   <td>0</td>
  </tr>
  <tr>
   <th>7</th>
   <td>Populations</td>
   <td>The number of individual solutions that are evolved on parallel </td>
   <td>50</td>
  </tr>
  <tr>
   <th>8</th>
   <td>Generations</td>
   <td>The maximum number of generations which we allow for the population to get evolved</td>
   <td>100</td>
  </tr>
  <tr>
   <th>9</th>
   <td>Mutation Probability</td>
   <td>The probability for applying Gaussian mutation operator</td>
   <td>0.01</td>
  </tr>
  <tr>
   <th>10</th>
   <td>Arithmetic Crossover Probability</td>
   <td>The probability (multiplied by 100) for using arithmetic crossover operator</td>
   <td>0</td>
  </tr>
  <tr>
   <th>11</th>
   <td>Two Points Crossover Probability</td>
   <td>The probability (multiplied by 100) for using two-point crossover operator</td>
   <td>0.9</td>
  </tr>
  <tr>
   <th>12</th>
   <td>Number of Folds</td>
   <td>The number of k folds in k-fold CV</td>
   <td>Default is 5. Examples use 10.</td>
  </tr>
  <tr>
   <th>13</th>
   <td>Filetype</td>
   <td>Input dataset's filetype</td>
   <td>8</td>
  </tr>
  <tr>
   <th>14</th>
   <td>Has Feature Headers</td>
   <td>0 if the input ds doesn't have feature headers, 1 if it has.</td>
   <td>1</td>
  </tr>
  <tr>
   <th>15</th>
   <td>Has Sample Headers</td>
   <td>0 if the input ds doesn't have sample headers, 1 if it has.</td>
   <td>1</td>
  </tr>
  <tr>
   <th>16</th>
   <td>Output Directory</td>
   <td>String with the desired output folder directory</td>
   <td>string w/ location</td>
  </tr>



</table>

‘ppi_training_dataset.txt’, ‘dg_dataset_t.csv’ are the transposed versions of the preprocessed PPI and AFFINITY training datasets. ‘ppi_training_labelss.txt’, ‘dg_labels.txt’ are the sample labels of the said datasets.

All these files are contained in the directory <code>example_datasets</code> > <code>cmd_input_datasets</code>.

Examples from the output folders from training and testing results of this project can be found in the directory <code>model_results</code>.

**The Unix commands for testing the classification models are:**

For the Ensemble method:
```
sudo python3 testing_multibiomarker_predictive_analytics_model_backend.py /Downloads/test_ds_preprocess_t.csv /Downloads/test_ds_labels.txt "" "" "" /Downloads/ Output_folder_classifier /features_list.txt 2 1 /Downloads/models_check_1/models_1.zip 2 1 "" 8 1 0 /Output_folder_classifier /training_labels.txt "" /Downloads/ Output_folder_classifier/length_of_features_from_training.txt Output_folder_testing_classifier/ ""
```

**Testing Algorithm Parameters:**
<table>
  <tr>
   <th>Position</th>
   <th>Parameter</th>
   <th>Description</th>
   <th>Default value</th> 
  </tr>
  <tr>
   <th>0</th>
   <td>Dataset Filename</td>
   <td>Input test dataset that has features as rows and samples as columns</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>1</th>
   <td>Labels Filename</td>
   <td>Tab delimited txt file with test label values</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>2</th>
   <td>Maximums Filename</td>
   <td>String with file of the maximum values of the testset features</td>
   <td>NaN</td>
  </tr>
  <tr>
   <th>3</th>
   <td>Minimums Filename</td>
   <td>String with file of the minimum values of the testset features</td>
   <td>NaN</td>
  </tr>
  <tr>
   <th>4</th>
   <td>Averages Filename</td>
   <td>String with the average values from the preprocessing</td>
   <td>NaN</td>
  </tr>
  <tr>
   <th>5</th>
   <td>Features Filename</td>
   <td>The selected features filename from the training Output Folder</td>
   <td>String w/ location</td>
  </tr>
  <tr>
   <th>6</th>
   <td>Missing Imputation Method</td>
   <td>The missing imputation method done in the preprocessing step of the training data (1,2)</td>
   <td>2</td>
  </tr>
  <tr>
   <th>7</th>
   <td>Normalization Method</td>
   <td>The normalization method done in the preprocessing step of the training data (1,2)</td>
   <td>1</td>
  </tr>
  <tr>
   <th>8</th>
   <td>Model Filename</td>
   <td>The .zip folder containing the trained models for testing, the classifier chain and their selected features</td>
   <td>String w/ location</td>
  </tr>
  <tr>
   <th>9</th>
   <td>Selection Flag</td>
   <td>Integer indicating the type of model analysis</td>
   <td>0 for multiclass classification, 1 for regression, 2 for two-class classification</td>
  </tr>
  <tr>
   <th>10</th>
   <td>Data Preprocessing Flag</td>
   <td>1 if the testing data has been preprocessed, 0 if it hasn't</td>
   <td>1</td>
  </tr>
  <tr>
   <th>11</th>
   <td>Selected Comorbidities String</td>
   <td>The string with the selected comorbidities names, separated by commas or newline characters from the training phase</td>
   <td>NaN</td>
  </tr>
  <tr>
   <th>12</th>
   <td>Filetype</td>
   <td>7 if it is a gene expression file, any other integer if it's a biomarkers files</td>
   <td>8</td>
  </tr>
  <tr>
   <th>13</th>
   <td>Has Feature Headers</td>
   <td>0 if the input ds doesn't have feature headers, 1 if it has.</td>
   <td>1</td>
  </tr>
  <tr>
   <th>14</th>
   <td>Has Sample Headers</td>
   <td>0 if the input ds doesn't have sample headers, 1 if it has.</td>
   <td>0</td>
  </tr>
  <tr>
   <th>15</th>
   <td>Training Labels Filename</td>
   <td>The training labels filename from the training Output Folder</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>16</th>
   <td>Length of Features Filename</td>
   <td>The length of training features filename</td>
   <td>NaN</td>
  </tr>
  <tr>
   <th>17</th>
   <td>Length of Features from Training Filename</td>
   <td>The length of features file from the training Output Folder</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>18</th>
   <td>Output Folder</td>
   <td>String with the desired output folder directory</td>
   <td>string w/ location</td>
  </tr>
  <tr>
   <th>19</th>
   <td>Selected Comorbidities String</td>
   <td>The string with the selected comorbidities names, separated by commas or newline characters from the training phase</td>
   <td>NaN</td>
  </tr>
</table>

‘test_ds_preprocess_t.csv’ is the transposed version of the PPI dataset. ‘test_ds_labels.txt’, contains the sample labels of the said dataset. All these files are contained in the <code>example_datasets</code> > <code>cmd_input_datasets</code> directory.

The <code>models_1.zip</code> contains the best scoring models, the classifier chain file and the selected features from the corresponding models. All the information about these files is contained in the model training output folders contained in the <code>model_results</code> directory, while the zip files themselves can be found in the <code>model_results</code> > <code>zip_test_files</code> directory.

For the Best Model method:
The same as the Ensemble method, but instead of many .pkl files, models_1.zip contains a single best model.

## 3. BML Analysis

The files needed for the training and testing of the Benchmark Machine Learning (BML) Analysis models are contained in the <code>BML Analysis</code> directory.

The results and the code for training the models are presented in the following notebook:
```
BML_Final.ipynb
```

The datasets needed for running are ‘TEST_DATASET_PROCESSED.csv’ and ‘train_ds.csv’ contained in the same directory.
The trained model is saved as ‘train_full_features.json’



