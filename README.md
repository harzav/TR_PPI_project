# TR-PPI project

This is the directory containing the necessary files and codes for running the classification and regression models of the TR-PPI project. 
This is a project aiming at PPI discovery through an Evolutionary Optimization Algorithm and quantification of the strength of PPIs through regression models based on an HDOCKlite endpoint. 
These models are to be used on the TR interactome, with the purpose of finding potential novel TR interactions and filtering the most probable ones.

The Flowchart of the proposed method and the Tutorial for the feature calculation and model creation are presented below.

<h2>A. Flowchart of the Proposed Method</h2>

![Rusell’s Negative Dataset (1)](https://github.com/harzav/PPI_project/assets/165158954/db438aee-6623-449a-bac4-f38e62dcdb55)

<h2>B. Tutorial for model creation </h2>

<h4>1.	Creation of the training datasets- feature calculation</h4>

The code for calculating the features for all Datasets in the project (affinity test and training datasets, PPI test and training datasets and taste receptor dataset) can be found in the directory <code>codes</code>.
‘feature_calculation_code.py’

Receives as input: A csv dataset with 4 columns that must be named ‘uidA’, ‘uidB’, protein_accession_A’, ‘protein_accession_B’. It can have more columns but these 4 must be present.

 ![image](https://github.com/harzav/PPI_project/assets/165158954/43e62396-63e0-406f-8f1b-88f0257e1961)

Output: A csv dataset with 61 additional features named:  'BP_similarity', 'MF_similarity', 'CC_similarity', 'Exists in MINT?', 'Exists in DIP?', 'Exists in APID?','Exists in BIOGRID?', 'Sequence_similarity', 'pfam_interaction','MW dif', 'Aromaticity dif', 'Instability dif', 'helix_fraction_dif', 'turn_fraction_dif', 'sheet_fraction_dif', 'cys_reduced_dif', 'gravy_dif', 'ph7_charge_dif', 'A %', 'L %', 'F %', 'I %', 'M %', 'V %', 'S %', 'P %', 'T %', 'Y %', 'H %', 'Q %', 'N %', 'K %', 'D %', 'E %', 'C %', 'W %', 'R %', 'G %',   'GSE227375_spearman', 'GSE228702_spearman', ‘0, 1, 2...14’, ‘Homologous in Mouse/ Drosophila/Yeast/Ecoli ’,  ‘Subcellular Co-localization?’

Output examples from the different datasets created for this project can be found in the <code>example_datasets</code> directory (AFFINITY_DATASET, PPI_TESTING_DATASET,PPI_TRAINING_DATASET)

<code>Datasets</code>: The directory containing all the datasets needed for the calculation of the features. 

**! Important !**

For the creation of the regression models, the HDOCKlite endpoint is necessary. For the calculation of this endpoint (and additionally the MMGBSA endpoint) run the Python script:

```
HDOCK_MMGBSA.py
```

located in the <code>HDOCK_MMGBSA_calculation</code> directory along with the supporting README file.

<h4>2.	Model Creation and Testing </h4>

**The scripts for training and testing the models can be found in the directory <code>codes</code>.**

For training:
```
biomarker_discovery_script_selection_backend.py
```

For testing:
```
testing_multibiomarker_predictive_analytics_model_backend.py
```

**The commands for training the models are:**

For the Binary Classifier:
```
sudo python3 biomarker_discovery_script_selection_backend.py ~/Downloads/ppi_training_dataset.txt ~/Downloads/ppi_training_labelss.txt "1,10,10,1,1,1,1,1" "" 2 0 0 20 50 0.01 0 0.9 10 8 1 1 ~/Downloads/Output_folder_classifier/
```

For the Regressor:
```
sudo python3 biomarker_discovery_script_selection_backend.py ~/Downloads/new_affinity_t.csv ~/Downloads/new_affinity_labels.txt "1,10,1,1,1,1,1,1" "" 1 0 0 50 1000 0.01 0 0.9 10 8 1 1 ~/Downloads/Output_folder_regressor/
```

‘ppi_training_dataset.txt’, ‘new_affinity_t.csv’ are the transposed versions of the preprocessed PPI and AFFINITY training datasets. ‘ppi_training_labelss.txt’, ‘new_affinity_labels.txt’ are the sample labels of the said datasets.

All these files are contained in the directory <code>example_datasets</code> > <code>cmd_input_datasets</code>.

Examples from the output folders from training results of this project can be found in the directory <code>model_results</code>.

**The commands for testing the classification models are:**

For the Ensemble method:
```
sudo python3 testing_multibiomarker_predictive_analytics_model_backend.py /Downloads/test_ds_preprocess_t.csv /Downloads/test_ds_labels.txt "" "" "" "" /Downloads/ Output_folder_classifier /features_list.txt 2 1 /Downloads/models_check_1/models_1.zip 2 1 "" 8 1 0 /Output_folder_classifier /training_labels.txt "" /Downloads/ Output_folder_classifier/length_of_features_from_training.txt Output_folder_testing_classifier/ ""
```

‘test_ds_preprocess_t.csv’ is the transposed version of the PPI dataset. ‘test_ds_labels.txt’, contains the sample labels of the said dataset. All these files are contained in the <code>example_datasets</code> > <code>cmd_input_datasets</code> directory.

The <code>models_1.zip</code> contains the best scoring models, the classifier chain file and the selected features from the corresponding models. All the information about these files is contained in the model training output folders contained in the <code>model_results</code> directory, while the zip files themselves can be found in the <code>model_results</code> > <code>zip_test_files</code> directory.

For the Best Model method:
The same as the Ensemble method, but instead of many .pkl files, models_1.zip contains a single best model.

<h4>3.	BML Analysis</h4>

The files needed for the training and testing of the Benchmark Machine Learning (BML) Analysis models are contained in the <code>BML Analysis</code> directory.

The results and the code for training the models are presented in the following notebook:
```
BML_Final.ipynb
```

The datasets needed for running are ‘TEST_DATASET_PROCESSED.csv’ and ‘train_ds.csv’ contained in the same directory.
The trained model is saved as ‘train_full_features.json’



