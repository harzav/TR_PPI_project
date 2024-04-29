# TR-PPI project

This is the directory containing the necessary files and codes for running the classification and regression models of the TR-PPI project. 
This is a project aiming at PPI discovery through an Evolutionary Optimization Algorithm and quantification of the strength of PPIs through regression models based on an HDOCKlite endpoint. 
These models are to be used on the TR interactome, with the purpose of finding potential novel TR interactions and filtering the most probable ones.

The Flowchart of the proposed method and the Tutorial for the feature calculation and model creation are presented below.

<h2>A. Flowchart of the Proposed Method</h2>

![Rusell’s Negative Dataset (1)](https://github.com/harzav/PPI_project/assets/165158954/db438aee-6623-449a-bac4-f38e62dcdb55)

<h2>B. Tutorial for model creation </h2>

<h4>1.	Creation of the training datasets- feature calculation</h4>

The code for calculating the features for all Datasets in the project (affinity test and training datasets, PPI test and training datasets and taste receptor dataset) can be found on the directory ‘codes’.
‘feature_calculation_code.py’

Receives as input: A csv dataset with 4 columns that must be named ‘uidA’, ‘uidB’, protein_accession_A’, ‘protein_accession_B’. It can have more columns but these 4 must be present.
 
Output: A csv dataset with 61 additional features named:  'BP_similarity', 'MF_similarity', 'CC_similarity', 'Exists in MINT?', 'Exists in DIP?', 'Exists in APID?','Exists in BIOGRID?', 'Sequence_similarity', 'pfam_interaction','MW dif', 'Aromaticity dif', 'Instability dif', 'helix_fraction_dif', 'turn_fraction_dif', 'sheet_fraction_dif', 'cys_reduced_dif', 'gravy_dif', 'ph7_charge_dif', 'A %', 'L %', 'F %', 'I %', 'M %', 'V %', 'S %', 'P %', 'T %', 'Y %', 'H %', 'Q %', 'N %', 'K %', 'D %', 'E %', 'C %', 'W %', 'R %', 'G %',   'GSE227375_spearman', 'GSE228702_spearman', ‘0, 1, 2...14’, ‘Homologous in Mouse/ Drosophila/Yeast/Ecoli ’,  ‘Subcellular Co-localization?’
Output examples from the different datasets created for this project can be found on the ‘example_datasets’ directory (AFFINITY_DATASET, PPI_TESTING_DATASET,PPI_TRAINING_DATASET)

‘.Datasets’ : The directory containing all the datasets needed for the calculation of the features. 

