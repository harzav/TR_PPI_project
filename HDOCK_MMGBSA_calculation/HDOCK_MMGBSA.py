import pandas as pd
import requests
import os
import re
import subprocess
import shutil
import sys

### Function to download a CIF file from AlphaFold ###
def download_cif_file(entry_id):
    # Create the CIF_FILES folder if it doesn't exist
    folder_name = "CIF_FILES"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    url = f"https://alphafold.ebi.ac.uk/files/AF-{entry_id}-F1-model_v4.cif"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = os.path.join(folder_name, f"{entry_id}.cif")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Successfully downloaded CIF file: {file_path}")
        else:
            print(f"Failed to download CIF file: {entry_id}.cif. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading CIF file for entry_id: {entry_id}")
        print(f"Error message: {str(e)}")

### Function to download a PDB file from AlphaFold ###
def download_pdb_file(entry_id):
    # Create the PDB_FILES folder if it doesn't exist
    folder_name = "PDB_FILES"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    url = f"https://alphafold.ebi.ac.uk/files/AF-{entry_id}-F1-model_v4.pdb"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = os.path.join(folder_name, f"{entry_id}.pdb")
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"Successfully downloaded PDB file: {file_path}")
        else:
            print(f"Failed to download PDB file: {entry_id}.pdb. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading PDB file for entry_id: {entry_id}")
        print(f"Error message: {str(e)}")

### Function to perform protein-protein docking using HDOCK ###
def perform_docking(entry_id_A, entry_id_B):
    # Define the paths for the receptor and ligand PDB files
    pdb_file_A = f"PDB_FILES/{entry_id_A}.pdb"
    pdb_file_B = f"PDB_FILES/{entry_id_B}.pdb"

    # Create the "HDOCK_Outputs" directory if it doesn't exist
    if not os.path.exists("HDOCK_Outputs"):
        os.makedirs("HDOCK_Outputs")
    
    # Change the current working directory to "HDOCK_Outputs"
    os.chdir("HDOCK_Outputs")

    # Check if the complex_models_dir already exists
    complex_models_dir = f"{entry_id_A}_{entry_id_B}_complex_models"
    if os.path.exists(complex_models_dir):
        print(f"Complex models directory '{complex_models_dir}' already exists. Skipping docking.")

        # Change the current working directory back to the original directory
        os.chdir("..")
        
        return f"HDOCK_Outputs/{complex_models_dir}"  # Return the path to the existing directory

    # Define the output file for HDOCKlite results
    hdock_output_file = f"{entry_id_A}_{entry_id_B}_Hdock.out"

    # Run HDOCKlite ab initio docking
    print(f"Starting HDOCK docking of the '{entry_id_A}' and '{entry_id_B}' pdb files.")
    hdock_command = f".././hdock ../{pdb_file_A} ../{pdb_file_B} -out {hdock_output_file}"
    subprocess.run(hdock_command, shell=True)

    # Run createpl to generate complex models from HDOCKlite solutions (total models generated: 10)
    createpl_command = f".././createpl {hdock_output_file} {complex_models_dir} -nmax 10 -complex -models"
    subprocess.run(createpl_command, shell=True)

    # Create a directory to store the complex models
    os.makedirs(complex_models_dir, exist_ok=True)

    # Move the necessary files to the complex_models_dir
    os.rename(hdock_output_file, os.path.join(complex_models_dir, hdock_output_file))
    for model_num in range(1, 11):
        model_pdb_file = f"model_{model_num}.pdb"
        os.rename(model_pdb_file, os.path.join(complex_models_dir, model_pdb_file))

    # Change the current working directory back to the original directory
    os.chdir("..")

    return f"HDOCK_Outputs/{complex_models_dir}"  # Return the path to the directory containing complex models

### Function to parse the HDOCK docking result and extract the binding affinity of the first 10 models ###
def parse_docking_result(output_dir, number_models_average):
    docking_scores = []
    number_models_average = number_models_average + 1
    
    # Iterate on the first "number_models_average" models
    for model_num in range(1, number_models_average):
        model_pdb_file = os.path.join(output_dir, f"model_{model_num}.pdb")
        with open(model_pdb_file, 'r') as f:
            lines = f.readlines()

        # Collect the docking score values of each model    
        for line in lines:
            if line.startswith("REMARK Score:"):
                docking_score = float(line.split(":")[1].strip())
                docking_scores.append(docking_score)
                break
    
    # Average or the docking score values
    average_docking_score = sum(docking_scores) / len(docking_scores)
    return average_docking_score

### Function to perform MMGBSA for binding affinity calculation ###
def perform_MMGBSA(entry_id_A, entry_id_B, nt, gpu_id):
    # Create the "MMGBSA_Outputs" directory if it doesn't exist
    if not os.path.exists("MMGBSA_Outputs"):
        os.makedirs("MMGBSA_Outputs")
    
    # Change the current working directory to "MMGBSA_Outputs"
    os.chdir("MMGBSA_Outputs")
    
    # Define the paths for the mmgbsa_models_dir and other required files
    mmgbsa_models_dir = f"{entry_id_A}_{entry_id_B}_complex_models"

    # Check if the mmgbsa_models_dir exists
    if not os.path.exists(mmgbsa_models_dir):

        # Copy the docked files from HDOCK_Outputs to MMGBSA_Outputs
        source_dir = f"../HDOCK_Outputs/{entry_id_A}_{entry_id_B}_complex_models"
        shutil.copytree(source_dir, mmgbsa_models_dir)
    else:
        print(f"MMGBSA already performed for the {mmgbsa_models_dir}")
        
        # Change the current working directory back to the original directory
        os.chdir("..")
        
        return f"MMGBSA_Outputs/{mmgbsa_models_dir}"  # Return the path to the directory containing mmgbsa calculations
    
    input_dir = "../MD_MMGBSA_INPUT"
    required_files = ["em_mmgbsa.sh", "input.in"]
    required_dirs = ["mdp_AA"]

    for file in required_files:
        shutil.copy(os.path.join(input_dir, file), os.path.join(mmgbsa_models_dir, file))

    for directory in required_dirs:
        source_dir = os.path.join(input_dir, directory)
        target_dir = os.path.join(mmgbsa_models_dir, directory)
        shutil.copytree(source_dir, target_dir)

    # Change the current working directory to the mmgbsa_models_dir
    os.chdir(mmgbsa_models_dir)
    
    # Execute the "mdAAsolv.sh" script with the specified arguments
    mdAAsolv_script = "./em_mmgbsa.sh"
    subprocess.run([mdAAsolv_script, str(nt), str(gpu_id)])

    # Change the current working directory back to the original directory
    os.chdir("../..")
    
    return f"MMGBSA_Outputs/{mmgbsa_models_dir}"  # Return the path to the directory containing mmgbsa calculations

### Function to parse MMGBSA results ###
def parse_MMGBSA_result(folder_path):
    # Define the paths for the mmgbsa_models_dir and other required files
    mmgbsa_models_dir = f"{entry_id_A}_{entry_id_B}_complex_models"
    
    # Change the current working directory to the mmgbsa_models_dir_path
    os.chdir(os.path.join(folder_path, mmgbsa_models_dir))

    # Read the MMGBSA_10_sorted.dat file and extract the 10 numbers
    with open("MMGBSA_10_sorted.dat", "r") as f:
        MMGBSA_results = [float(line.strip()) for line in f.readlines()]

    # Change the current working directory back to the original directory
    os.chdir("../..")

    return MMGBSA_results # Return the list of 10 numbers

#############
### Start ###
#############

# Read CSV file and process entries using pandas
csv_filename = "FINAL_DATASET.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_filename)

# Use a boolean variable to control whether the functions should be activated or deactivated
function_pdb_cif_active = False           # True or False
function_docking_active = False           # True or False
function_binding_active = False           # True or False
function_write_binding_csv_active = True  # True or False

##################################################################################################
### Download PDB and CIF files for each entry in the CSV file and print the confidence numbers ###
##################################################################################################

# Use an if statement to check the condition and call the function accordingly
if function_pdb_cif_active:
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        entry_id_A = row["uidA"]
        entry_id_B = row["uidB"]

        # Download CIF and PDB files
        download_cif_file(entry_id_A)
        download_pdb_file(entry_id_A)
        download_cif_file(entry_id_B)
        download_pdb_file(entry_id_B)

        # Process CIF file and extract the confidence number of protein A and B
        cif_filename_A = f"CIF_FILES/{entry_id_A}.cif"
        cif_filename_B = f"CIF_FILES/{entry_id_B}.cif"
        target_string = "_ma_qa_metric_global.metric_value"

        # Confidence number of protein A
        try:
            with open(cif_filename_A, "r") as cif_file:
                cif_content = cif_file.read()

            pattern = fr"{re.escape(target_string)}\s+(-?\d+\.\d+)"
            match = re.search(pattern, cif_content)

            if match:
                number = float(match.group(1))
                print(f"Found the target string with number: {number}")
            else:
                number = None
                print("Target string not found in the CIF file.")

            # Assign the values to new columns in the DataFrame
            df.at[index, "ConfidenceA"] = number

        except FileNotFoundError:
            print(f"CIF file not found for entry_id: {entry_id_A}")
            continue

        # Confidence number of protein B
        try:
            with open(cif_filename_B, "r") as cif_file:
                cif_content = cif_file.read()

            pattern = fr"{re.escape(target_string)}\s+(-?\d+\.\d+)"
            match = re.search(pattern, cif_content)

            if match:
                number = float(match.group(1))
                print(f"Found the target string with number: {number}")
            else:
                number = None
                print("Target string not found in the CIF file.")

            # Assign the values to new columns in the DataFrame
            df.at[index, "ConfidenceB"] = number

        except FileNotFoundError:
            print(f"CIF file not found for entry_id: {entry_id_B}")
            continue

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_filename, index=False)
    print("CSV file updated with Confidence of Proteins A and B.")

###########################################
### Calculate Docking score using HDOCK ###
###########################################

# Use an if statement to check the condition and call the function accordingly
if function_docking_active:
    # Set confidence threshold (0-100) for the pdb files used in docking procedure
    confidence_threshold = 80.0

    # Set the number of pdb models for the average docking score calculation (1-10)
    number_models = 1

    # Iterate over each row in the DataFrame
    result_rows = []
    for index, row in df.iterrows():
        entry_id_A = row["uidA"]
        entry_id_B = row["uidB"]

        # Check if docking can be performed based on confidence numbers
        if pd.notnull(row['ConfidenceA']) and pd.notnull(row['ConfidenceB']) and row['ConfidenceA'] > confidence_threshold and row['ConfidenceB'] > confidence_threshold:
            # Perform docking procedure using the downloaded PDB files and HDOCK
            complex_models_dir_path = perform_docking(entry_id_A, entry_id_B)

            # Calculate the average docking score for the first 10 models
            average_score = parse_docking_result(complex_models_dir_path, number_models)

            # Add a new row to the result DataFrame
            result_rows.append([entry_id_A, entry_id_B, row['ConfidenceA'], row['ConfidenceB'], row['PPI_type'], average_score])

    # Save the result to DataFrame and then to a new CSV file
    result_df = pd.DataFrame(result_rows, columns=["uidA", "uidB", "ConfidenceA", "ConfidenceB", "PPI_type", "Average_Docking_Score"])
    result_csv_filename = "dataset_1.csv"
    result_df.to_csv(result_csv_filename, index=False)
    print(f"New CSV file '{result_csv_filename}' created with docking results.")

###############################################
### Calculate binding affinity using MMGBSA ###
###############################################
if function_binding_active:
    # Set confidence threshold (0-100) for the pdb files used in docking procedure
    confidence_threshold = 80.0

    # Set the number of pdb models to perform docking calculation (1-10)
    number_models = 10

    # Set NT (n° cores to use in GROMACS) and GPU_ID (n° gpu to use in GROMACS) parameters 
    nt = 10
    gpu_id = 0

    # Iterate over each row in the DataFrame
    result_rows = []
    for index, row in df.iterrows():
        entry_id_A = row["uidA"]
        entry_id_B = row["uidB"]

        # Check if binding affinity can be performed based on confidence numbers
        if pd.notnull(row['ConfidenceA']) and pd.notnull(row['ConfidenceB']) and row['ConfidenceA'] > confidence_threshold and row['ConfidenceB'] > confidence_threshold:
            # Perform MMGBSA binding affinity procedure using the docked files
            complex_models_dir_path = perform_MMGBSA(entry_id_A, entry_id_B, nt, gpu_id)

###############################################
### Calculate binding affinity using MMGBSA ###
###############################################
if function_write_binding_csv_active:
    # Set confidence threshold (0-100) for the pdb files used in docking procedure
    confidence_threshold = 80.0

    # Iterate over each row in the DataFrame
    result_rows = []
    for index, row in df.iterrows():
        entry_id_A = row["uidA"]
        entry_id_B = row["uidB"]

        # Check if binding affinity can be performed based on confidence numbers
        if pd.notnull(row['ConfidenceA']) and pd.notnull(row['ConfidenceB']) and row['ConfidenceA'] > confidence_threshold and row['ConfidenceB'] > confidence_threshold:
            # Perform MMGBSA binding affinity procedure using the docked files
            MMGBSA_models_dir_path = "./MMGBSA_Outputs"
            MMGBSA_results = parse_MMGBSA_result(MMGBSA_models_dir_path)

            # Add a new row to the result DataFrame
            result_rows.append([entry_id_A, entry_id_B, row['ConfidenceA'], row['ConfidenceB'], row['PPI_type']] + MMGBSA_results)
    
    # Save the result to DataFrame and then to a new CSV file
    result_df = pd.DataFrame(result_rows, columns=["uidA", "uidB", "ConfidenceA", "ConfidenceB", "PPI_type", "MMGBSA_1", "MMGBSA_2", "MMGBSA_3", "MMGBSA_4", "MMGBSA_5", "MMGBSA_6", "MMGBSA_7", "MMGBSA_8", "MMGBSA_9", "MMGBSA_10"])
    result_csv_filename = "dataset_MMGBSA.csv"
    result_df.to_csv(result_csv_filename, index=False)
    print(f"New CSV file '{result_csv_filename}' created with MMGBSA results.")