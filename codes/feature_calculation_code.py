import pandas as pd
import functools
from pygosemsim import term_set
from pygosemsim import annotation
from pygosemsim import similarity
from pygosemsim import graph
from goatools import obo_parser
import wget
import os
from Bio import Align
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import requests

#set output path
path= ''
#provide input csv ds
test_ds= pd. read_csv('input.csv')

#---------------------------------------------------------------------------------------#
#   E-value calculation
#---------------------------------------------------------------------------------------#

# Function to perform pairwise alignment
def pairwise_align(row):
    aligner = Align.PairwiseAligner(match_score=1.0)
    if pd.notnull(row['seq_A']) and pd.notnull(row['seq_B']):
        score = aligner.score(row['seq_A'], row['seq_B'])
        return score
    else:
        return 'NaN'

# Apply the function to create a new column 'X'
test_ds['X'] = test_ds.apply(pairwise_align, axis=1)

# Function to calculate mean length and 'e_val'
def calc_e_val(row):
    if pd.notnull(row['seq_A']) and pd.notnull(row['seq_B']):
        mean_len = (len(row['seq_A']) + len(row['seq_B'])) / 2
        if pd.notnull(row['X']):
            e_val = mean_len * (2.7**(-row['X']))
            return e_val
    return 'NaN'

# Apply the function to create a new column 'e_val'
test_ds['Sequence_similarity'] = test_ds.apply(calc_e_val, axis=1)
test_ds = test_ds.drop('X', axis=1)

#---------------------------------------------------------------------------------------#
#   Extra feature calculation
#---------------------------------------------------------------------------------------#

sequences1 = test_ds[['seq_A', 'seq_B']].apply(lambda x: (str(x['seq_A']).replace('X', ''), str(x['seq_B']).replace('X', '')), axis=1).tolist()

sequences = [[i[0].replace('U', ''), i[1].replace('U', '')] for i in sequences1]
sequences = [[i[0].replace(' ', ''), i[1].replace(' ', '')] for i in sequences]
sequences = [[i[0].replace('\n', ''), i[1].replace('\n', '')] for i in sequences]


def get_features(X):
    amino_acids = ['A', 'L', 'I', 'M', 'F', 'V', 'S', 'P', 'T', 'Y', 'H', 'Q', 'N', 'K', 'D', 'E', 'C', 'W', 'R', 'G']
    aaper = X.get_amino_acids_percent()
    features = [aaper[aa] for aa in amino_acids]

    mw = X.molecular_weight()
    arom = X.aromaticity()
    insta = X.instability_index()
    fraction = X.secondary_structure_fraction()
    helix_fraction, turn_fraction, sheet_fraction = fraction[0], fraction[1], fraction[2]
    extinct = X.molar_extinction_coefficient()
    cys_reduced, cys_residues = extinct[0], extinct[1]
    gravy = X.gravy(scale='KyteDoolitle')
    ph_charge = X.charge_at_pH(7)

    features.extend([mw, arom, insta, helix_fraction, turn_fraction, sheet_fraction,
                     cys_reduced, cys_residues, gravy, ph_charge])
    return features

# Calculate features for each sequence pair
feature_list = []
for seq in sequences:
    X0 = ProteinAnalysis(seq[0])
    seq0 = get_features(X0)
    X1 = ProteinAnalysis(seq[1])
    seq1 = get_features(X1)
    substr_list = [abs(x - y) for x, y in zip(seq0, seq1)]
    abs_list = [abs(val) for val in substr_list]
    feature_list.append(abs_list)

# Create DataFrame from the list of feature differences
column_names = ['A %', 'L %', 'F %', 'I %', 'M %', 'V %', 'S %', 'P %', 'T %', 'Y %', 'H %', 'Q %', 'N %', 'K %', 'D %', 'E %', 'C %', 'W %', 'R %', 'G %',
                'MW dif', 'Aromaticity dif', 'Instability dif', 'helix_fraction_dif', 'turn_fraction_dif', 'sheet_fraction_dif',
                'cys_reduced_dif', 'cys_residues_dif', 'gravy_dif', 'ph7_charge_dif']
new_df = pd.DataFrame(feature_list, columns=column_names)

# Concatenate new DataFrame with the original DataFrame
test_ds = pd.concat([test_ds, new_df], axis=1)

#---------------------------------------------------------------------------------------#
#   GO SIMILARITY CODE
#---------------------------------------------------------------------------------------#
G = graph.from_resource("go-basic")
similarity.precalc_lower_bounds(G)
annot = annotation.from_resource("goa_human")

test_list =[]
for index, rows in test_ds.iterrows():
    my_list =[rows.uidA, rows.uidB]
    test_list.append(my_list)

annot_list_full=[]
for list1 in test_list:
    annot_list=[]
    for i in list1:
        try:
            annot1 = str(annot[i]["annotation"].keys())
            annot2=annot1.strip('dict_keys([])')
            annot2=annot2.strip("'")
            annot3=list(annot2.split("', '"))
            annot_list.append(annot3)
        except KeyError:
            annot_list.append("NaN")
    annot_list_full.append(annot_list)

go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
data_folder = os.getcwd() + '/data'

if(not os.path.isfile(data_folder)):
    try:
        os.mkdir(data_folder)
    except OSError as e:
        if(e.errno != 17):
            raise e
else:
    raise Exception('Data path (' + data_folder + ') exists as a file. '
                'Please rename, remove or change the desired location of the data path.')

if(not os.path.isfile(data_folder+'/go-basic.obo')):
    go_obo = wget.download(go_obo_url, data_folder+'/go-basic.obo')
else:
    go_obo = data_folder+'/go-basic.obo'
go = obo_parser.GODag(go_obo)

def add_similarity_to_test(similarity_name, similarity_df_header):
    go_sim_list=[]
    for annot_list in annot_list_full:
        def filter_func(filter_term):
            BP_list_A=[]
            BP_list_B=[]
            BP_list=[BP_list_A,BP_list_B]
            if annot_list[0]!='NaN':
                for i in annot_list[0]:
                    go_id = i
                    try:
                        go_term = go[go_id]
                        if go_term.namespace == filter_term:
                            BP_list_A.append(i)
                    except KeyError:
                        continue
            else:
                BP_list_A.append('NaN')
            if annot_list[1]!='NaN':
                for i in annot_list[1]:
                    go_id = i
                    try:
                        go_term = go[go_id]
                        if go_term.namespace == filter_term:
                            BP_list_B.append(i)
                    except KeyError:
                        continue
            else:
                BP_list_B.append('NaN')
            if ((BP_list_A != 'NaN') & (BP_list_A != 'NaN')):
                sf = functools.partial(term_set.sim_func, G, similarity.lin)
                go_sim=term_set.sim_bma(BP_list[0], BP_list[1], sf)
                return go_sim
            else:
                return 'NaN'

        bp_sim= filter_func(similarity_name)
        go_sim_list.append(bp_sim)

    test_ds[similarity_df_header]=go_sim_list

add_similarity_to_test('biological_process', 'BP_similarity')
add_similarity_to_test('molecular_function', 'MF_similarity')
add_similarity_to_test('cellular_component', 'CC_similarity')

test_list2 =[]
for index, rows in test_ds.iterrows():
    my_list =[rows.BP_similarity , rows.MF_similarity, rows.CC_similarity]
    test_list2.append(my_list)

for i in test_list2:
    for l in range(len(i)):
        if (str(i[l])!= 'nan') :
            for j in range(len(i)):
                if str(i[j]) == 'nan':
                    i[j] = 0

test_ds['D']=test_list2

test_ds[['BP_similarity', 'MF_similarity', 'CC_similarity']] = test_ds.pop('D').tolist()



#---------------------------------------------------------------------------------------#
#  INTERPRO CODE
#---------------------------------------------------------------------------------------#

def fetch_pfam_ids(protein_list):
    unique_proteins = set(protein_list)
    pfam_ids = {}
    for prot in unique_proteins:
        url = f"https://www.ebi.ac.uk/interpro/api/entry/all/protein/UniProt/{prot}/"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                payload = response.json()
                if 'results' in payload and isinstance(payload['results'], list):
                    ipr_list = [item['metadata']['accession'] for item in payload['results'] if item.get('metadata', {}).get('source_database') == 'pfam']
                    pfam_ids[prot] = ipr_list if ipr_list else 'NaN'
                else:
                    pfam_ids[prot] = 'NaN'
            else:
                pfam_ids[prot] = 'NaN'
        except (requests.RequestException, ValueError):
            pfam_ids[prot] = 'NaN'
    return pfam_ids

def add_pfam_sim(test_ds):
    testA_list = test_ds.uidA.values.tolist()
    testB_list = test_ds.uidB.values.tolist()

    pfam_A = fetch_pfam_ids(testA_list)
    pfam_B = fetch_pfam_ids(testB_list)

    pfam_pair_list = [(pfam_A.get(prot, 'NaN'), pfam_B.get(prot, 'NaN')) for prot in testA_list]

    with open(".Datasets/3did_flat_Mar_4_2021.dat") as file:
        interact_list = [x.split("\t")[3:5] for x in file.readlines() if x.startswith('#=ID')]
        interact_list = [[p_A.split(".")[0], p_B.split(".")[0]] for p_A, p_B in interact_list]

    result_list = []
    for pfam_pair in pfam_pair_list:
        for interaction in interact_list:
            if (interaction[0] in pfam_pair[0] and interaction[1] in pfam_pair[1]) or (interaction[0] in pfam_pair[1] and interaction[1] in pfam_pair[0]):
                result_list.append(1)
            elif 'NaN' in pfam_pair:
                result_list.append('NaN')
            else:
                result_list.append(0)

    split_sp_lists = [result_list[x:x+len(interact_list)] for x in range(0, len(result_list), len(interact_list))]
    final_list = [1 if 1 in sublist else ('NaN' if 'NaN' in sublist else 0) for sublist in split_sp_lists]

    test_ds['pfam_interaction'] = final_list

add_pfam_sim(test_ds)

#---------------------------------------------------------------------------------------#
#  SL  CODE
#---------------------------------------------------------------------------------------#

# Load the data
map_ds = pd.read_csv(".Datasets/SL files/map_uniprot_human.csv", usecols=['ID', 'Protein_Accession'])
es_ds = pd.read_table(".Datasets/SL files/eSLDB_Homo_sapiens.txt", delimiter='\t', usecols=['Experimental annotation', 'SwissProt entry'])

# Clean and preprocess the data
es_ds = es_ds.dropna(subset=['Experimental annotation', 'SwissProt entry'])
es_ds.rename(columns={'Experimental annotation': 'Experimental_annotation', 'SwissProt entry': 'Protein_Accession'}, inplace=True)
es_ds['Experimental_annotation'] = es_ds['Experimental_annotation'].str.split(',')


# Extract unique pairs from test_ds
pair_list = list(map(tuple, test_ds[['uidA', 'uidB']].values))

values_list = []

for pair in pair_list:
    if pair[0] == pair[1]:
        values_list.append(1)
    else:
        match = map_ds[map_ds["ID"].isin(pair)]
        if len(match) > 0:
            merged = pd.merge(match, es_ds, on='Protein_Accession')
            annotations = merged['Experimental_annotation'].tolist()
            if len(annotations) == 1:
                values_list.append(annotations[0])  # Single annotation
            elif len(annotations) > 1:
                values_list.append(annotations[:2])  # First two annotations
            else:
                values_list.append('NaN')
        else:
            values_list.append('NaN')

# Define the function to compare annotations
def compare_annotations(annotation_list):
    if annotation_list == 1:
        return 1
    elif annotation_list == 'NaN':
        return 'NaN'
    else:
        if len(annotation_list) < 2:
            return 'NaN'
        annotations_A = set([annotation.strip() for annotation in annotation_list[0]])
        annotations_B = set([annotation.strip() for annotation in annotation_list[1]])
        if annotations_A.intersection(annotations_B):
            return 1
        else:
            return 0

# Define the function to process annotations
def process_annotations(val):
    if val == 1:
        return 1
    elif val == 'NaN':
        return 'NaN'
    else:
        return compare_annotations(val)

# Create the new column in test_ds
test_ds['Subcellular Co-localization?'] = [process_annotations(val) for val in values_list]

#---------------------------------------------------------------------------------------#
#   SPEARMAN  CODE
#---------------------------------------------------------------------------------------#

map_test=pd.read_csv(".Datasets/spearman files/map.csv")
map_test.rename(columns={'UniProtKB-AC':'UniprotID'}, inplace=True)
df1 = pd.read_table('.Datasets/spearman files/GDS181.soft', delimiter="\t", skiprows=[i for i in range(306,)])
df2 = pd.read_table('.Datasets/spearman files/GDS531.soft', delimiter="\t", skiprows=[i for i in range(210,)])
df3 = pd.read_table('.Datasets/spearman files/GDS807.soft', delimiter="\t", skiprows=[i for i in range(98,)], comment='!')
df4 = pd.read_table('.Datasets/spearman files/GDS841.soft', delimiter="\t", skiprows=[i for i in range(85,)], comment='!')
df5 = pd.read_table('.Datasets/spearman files/GDS806.soft', delimiter="\t", skiprows=[i for i in range(97,)], comment='!')
df6 = pd.read_table('.Datasets/spearman files/GDS987.soft', delimiter="\t", skiprows=[i for i in range(78,)])
df7 = pd.read_table('.Datasets/spearman files/GDS2855.soft', delimiter="\t", skiprows=[i for i in range(202,)])
df8 = pd.read_table('.Datasets/spearman files/GDS1088.soft', delimiter="\t", skiprows=[i for i in range(58,)])
df9 = pd.read_table('.Datasets/spearman files/GDS843.soft', delimiter="\t", skiprows=[i for i in range(113,)], comment='!')
df10 = pd.read_table('.Datasets/spearman files/GDS1402.soft', delimiter="\t", skiprows=[i for i in range(262,)])
df11 = pd.read_table('.Datasets/spearman files/GDS1085.soft', delimiter="\t", skiprows=[i for i in range(162,)], comment='!')
df12 = pd.read_table('.Datasets/spearman files/GDS534.soft', delimiter="\t", skiprows=[i for i in range(117,)])
df13 = pd.read_table('.Datasets/spearman files/GDS3257.soft', delimiter="\t", skiprows=[i for i in range(199,)])
df14 = pd.read_table('.Datasets/spearman files/GDS651.soft', delimiter="\t", skiprows=[i for i in range(78,)])
df15 = pd.read_table('.Datasets/spearman files/GDS596.soft', delimiter="\t", skiprows=[i for i in range(580,)])

GDtotal=[df1, df2, df3 , df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
GD_test=[]
for df in GDtotal:
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.iloc[2:].mean()))
    df = df.drop(columns=['ID_REF'])
    df = df.groupby('IDENTIFIER').mean().reset_index()
    GD_test.append(df)


def my_function(GD, test, map):
    import pandas as pd
    from scipy.stats import spearmanr
    
    uniprot_list =[]
    for index, rows in test.iterrows():
        my_list =[rows.uidA, rows.uidB]
        uniprot_list.append(my_list)

    map_list=[]
    for row in uniprot_list:
        mapds1=map[map["UniprotID"].isin(row) == True]
        map_list.append(mapds1)
    
    pair_list=[]
    for i in GD:
        for ds in map_list:
            GD1=i[i["IDENTIFIER"].isin(ds["ID"]) == True]
            pair_list.append(GD1)

    sp_list=[]
    for pair in pair_list:
        if len(pair.index)==2:
            pair=pair.drop(columns=['IDENTIFIER'])
            pair2=pair.transpose()
            rho, p= spearmanr(pair2)
            sp_list.append(rho)
        else:
            sp_list.append('nan')
    split_sp_lists = [sp_list[x:x+len(test.index)] for x in range(0, len(sp_list), len(test.index))]
    spearman_ds=pd.DataFrame(data= split_sp_lists)
    spearman_ds2=spearman_ds.transpose()

    final_ds=pd.concat([test, spearman_ds2], axis=1)

    return (final_ds)

test_ds= my_function(GD_test, test_ds, map_test)

test_ds.to_csv(path + 'Output1.csv', index=False)

#-------------------------------------------------------------------------------------------------------------#
# Calculate RNAseq features
#-------------------------------------------------------------------------------------------------------------#

# Load RNA dataset 1
expression_ds = pd.read_csv(".Datasets/extra rna seq columns/GSE227375_norm_counts_STAR.csv")
expression_ds.columns = expression_ds.columns.str.replace('Unnamed: 0', 'ENS_ID')

# Load mapping dataset
map_ds = pd.read_csv(".Datasets/extra rna seq columns/UID_to_ENSBL_mapping.csv", usecols=['uid', 'ID'])
map_ds['ID'] = map_ds['ID'].str.split('.').str[0]

# Map UIDs with ENSG
test_mapped = pd.merge(test_ds, map_ds, left_on='uidA', right_on='uid', how='left')
test_mapped.rename(columns={'ID': 'ID_A'}, inplace=True)
test_mapped = pd.merge(test_mapped, map_ds, left_on='uidB', right_on='uid', how='left')
test_mapped.rename(columns={'ID': 'ID_B'}, inplace=True)

# Process for ID_A
map_merge_A = pd.merge(test_mapped, expression_ds, left_on='ID_A', right_on='ENS_ID', how='left')
map_merge_A = map_merge_A.drop(columns=['uid_x', 'uid_y'])

# Calculate average expression values for ID_A
numeric_cols = map_merge_A.select_dtypes(include=[np.number]).columns
group_map_merge_A = map_merge_A.groupby('ENS_ID')[numeric_cols].mean()
group_map_merge_A['expression_list_A'] = group_map_merge_A.values.tolist()

map_merge_A = map_merge_A[['uidA', 'uidB', 'ENS_ID']]
merge_A = pd.merge(map_merge_A, group_map_merge_A, on='ENS_ID', how='left')
merge_A = merge_A[['uidA', 'uidB', 'expression_list_A']]

# Process for ID_B
map_merge_B = pd.merge(test_mapped, expression_ds, left_on='ID_B', right_on='ENS_ID', how='left')
map_merge_B = map_merge_B.drop(columns=['uid_x', 'uid_y'])

# Calculate average expression values for ID_B
group_map_merge_B = map_merge_B.groupby('ENS_ID')[numeric_cols].mean()
group_map_merge_B['expression_list_B'] = group_map_merge_B.values.tolist()

map_merge_B = map_merge_B[['uidA', 'uidB', 'ENS_ID']]
merge_B = pd.merge(map_merge_B, group_map_merge_B, on='ENS_ID', how='left')
merge_B = merge_B[['uidA', 'uidB', 'expression_list_B']]

# Drop duplicates and NaNs
merge_A = merge_A.drop_duplicates(subset=['uidA', 'uidB'])
merge_B = merge_B.drop_duplicates(subset=['uidA', 'uidB'])
merge_B = merge_B.drop(columns=['uidA', 'uidB'])

# Concatenate datasets
concatds = pd.concat([merge_A, merge_B], axis=1)

# Calculate Spearman correlation
exp_list = []
for index, rows in concatds.iterrows():
    my_list = [rows.expression_list_A, rows.expression_list_B]
    exp_list.append(my_list)

calc_list = []
for pair in exp_list:
    if (str(pair[0]) == 'nan') | (str(pair[1]) == 'nan'):
        calc_list.append('NaN')
    else:
        rho, p = spearmanr(pair[0], pair[1])
        calc_list.append(rho)

test_ds['GSE227375_spearman'] = calc_list

test_ds.to_csv('c:\\users\\harry\\desktop\\Output2.csv', index=False)

# Load expression dataset
expression_ds = pd.read_csv(".Datasets/extra rna seq columns/GSE228702_adjusted_expression_Cellcounts_granulatorAbis0_nnls.csv")
expression_ds.columns = expression_ds.columns.str.replace('Unnamed: 0', 'ENS_ID')

# Load mapping dataset
map_ds = pd.read_csv(".Datasets/extra rna seq columns/UID_to_GeneCards_mapping.csv", usecols=['uid', 'ID'])
map_ds['ID'] = map_ds['ID'].str.split('.').str[0]

# Map UIDs with GeneCards
test_mapped = pd.merge(test_ds, map_ds, left_on='uidA', right_on='uid', how='left')
test_mapped.rename(columns={'ID': 'ID_A'}, inplace=True)
test_mapped = pd.merge(test_mapped, map_ds, left_on='uidB', right_on='uid', how='left')
test_mapped.rename(columns={'ID': 'ID_B'}, inplace=True)

# Process for ID_A
map_merge_A = pd.merge(test_mapped, expression_ds, left_on='ID_A', right_on='ENS_ID', how='left')
map_merge_A = map_merge_A.drop(columns=['uid_x', 'uid_y'])

# Calculate average expression values for ID_A
numeric_cols = map_merge_A.select_dtypes(include=[np.number]).columns
group_map_merge_A = map_merge_A.groupby('ENS_ID')[numeric_cols].mean()
group_map_merge_A['expression_list_A'] = group_map_merge_A.values.tolist()

map_merge_A = map_merge_A[['uidA', 'uidB', 'ENS_ID']]
merge_A = pd.merge(map_merge_A, group_map_merge_A, on='ENS_ID', how='left')
merge_A = merge_A[['uidA', 'uidB', 'expression_list_A']]

# Process for ID_B
map_merge_B = pd.merge(test_mapped, expression_ds, left_on='ID_B', right_on='ENS_ID', how='left')
map_merge_B = map_merge_B.drop(columns=['uid_x', 'uid_y'])

# Calculate average expression values for ID_B
group_map_merge_B = map_merge_B.groupby('ENS_ID')[numeric_cols].mean()
group_map_merge_B['expression_list_B'] = group_map_merge_B.values.tolist()

map_merge_B = map_merge_B[['uidA', 'uidB', 'ENS_ID']]
merge_B = pd.merge(map_merge_B, group_map_merge_B, on='ENS_ID', how='left')
merge_B = merge_B[['uidA', 'uidB', 'expression_list_B']]

# Drop duplicates and NaNs
merge_A = merge_A.drop_duplicates(subset=['uidA', 'uidB'])
merge_B = merge_B.drop_duplicates(subset=['uidA', 'uidB'])
merge_B = merge_B.drop(columns=['uidA', 'uidB'])

# Concatenate datasets
concatds = pd.concat([merge_A, merge_B], axis=1)

# Calculate Spearman correlation
exp_list = []
for index, rows in concatds.iterrows():
    my_list = [rows.expression_list_A, rows.expression_list_B]
    exp_list.append(my_list)

calc_list = []
for pair in exp_list:
    if (str(pair[0]) == 'nan') | (str(pair[1]) == 'nan'):
        calc_list.append('NaN')
    else:
        rho, p = spearmanr(pair[0], pair[1])
        calc_list.append(rho)

test_ds['GSE228702_spearman'] = calc_list

test_ds.to_csv('c:\\users\\harry\\desktop\\Output3.csv', index=False)

#-------------------------------------------------------------------------------------------------------------#
# Calculate homology & database features
#-------------------------------------------------------------------------------------------------------------#

homologene_data=pd.read_csv(".Datasets/homology files/homologene_data.csv")
#----------remove .1, .2 at the end of protein accession Nos-----####
homologene_data['protein_accession'] = homologene_data['protein_accession'].str.split('.').str[0]

#import mouse
map_mouse=pd.read_csv(".Datasets/homology files/map_mouse.csv")
map_mouse['Protein_Accession'] = map_mouse['Protein_Accession'].str.split('.').str[0]
mouse_ppi=pd.read_csv(".Datasets/homology files/mouse_ppi.csv")
mouse_ppi.rename(columns={'ID interactor A':'uniprotid_A', 'ID interactor B':'uniprotid_B'}, inplace=True)

#import drosophila
map_drosophila=pd.read_csv(".Datasets/homology files/map_drosophila.csv")
map_drosophila['Protein_Accession'] = map_drosophila['Protein_Accession'].str.split('.').str[0]
dros_ppi=pd.read_csv(".Datasets/homology files/drosophila_ppi.csv")
dros_ppi.rename(columns={'ID interactor A':'uniprotid_A','ID interactor B':'uniprotid_B'}, inplace=True)
#import yeast
map_yeast=pd.read_csv(".Datasets/homology files/map_yeast.csv")
map_yeast['Protein_Accession'] = map_yeast['Protein_Accession'].str.split('.').str[0]
yeast_ppi=pd.read_csv(".Datasets/homology files/yeast_ppi.csv")
yeast_ppi.rename(columns={'ID interactor A':'uniprotid_A', 'ID interactor B':'uniprotid_B'}, inplace=True)
#import Ecoli
map_ecoli=pd.read_csv(".Datasets/homology files/map_Ecoli.csv")
map_ecoli['Protein_Accession'] = map_ecoli['Protein_Accession'].str.split('.').str[0]
ecoli_ppi=pd.read_csv(".Datasets/homology files/Ecoli_ppi.csv")
ecoli_ppi.rename(columns={'ID interactor A':'uniprotid_A', 'ID interactor B':'uniprotid_B'}, inplace=True)

test_list =[]
  
for index, rows in test_ds.iterrows():
    my_list =[rows.protein_accession_A, rows.protein_accession_B]
    test_list.append(my_list)


#Create Homologous map list

homol_list=[]
for pair in test_list:
    hom1=homologene_data[homologene_data["protein_accession"].isin(pair) == True]
    homol_list.append(hom1)
homol_list2=[]
for i in homol_list:
    hom1=homologene_data[homologene_data["HID"].isin(i['HID']) == True]
    homol_list2.append(hom1)

#--------FUNCTION HOMOLOGY PAIRS------#

def add_column_function(map_ds, ppi_ds,taxid_no, taxid_name):

    homol_list_filtered=[]
    for i in homol_list2:
        hom1=i.loc[i['taxid']==taxid_no]
        homol_list_filtered.append(hom1)
    merge_list=[]
    for ds in homol_list_filtered:
        mrg= pd.merge ( ds,map_ds, left_on='protein_accession', right_on='Protein_Accession')
        merge_list.append(mrg)

    pair_list=[]
    nan_list=['NaN','NaN']
    for i in merge_list:
        if i['HID'].nunique()==2:
            sublist=list(i['UniprotID'])
            pair_list.append(sublist)
        else:
            pair_list.append(nan_list)

    ppi_list=[]
    for index, rows in ppi_ds.iterrows():
        my_list =[rows.uniprotid_A, rows.uniprotid_B]
        ppi_list.append(my_list)
    
    value_list=[]
    for i in pair_list:
        for j in ppi_list:
            if (len(i)>=2)& (set(i)==set(j)):
                value_list.append(1)
            elif 'NaN' in i:
               value_list.append('NaN') 
            else:
                value_list.append(0)

    split_sp_lists = [value_list[x:x+len(ppi_list)] for x in range(0, len(value_list), len(ppi_list))]
    final_list=[]
    for i in split_sp_lists:
        if 1 in i:
            final_list.append(1)
        elif 'NaN' in i:
            final_list.append('NaN')
        else:
            final_list.append(0)

    test_ds[taxid_name ]=final_list

add_column_function(map_mouse, mouse_ppi, 10090, 'Homologous in Mouse')
add_column_function(map_drosophila, dros_ppi, 7227, 'Homologous in Drosophila')
add_column_function(map_yeast, yeast_ppi, 559292, 'Homologous in Yeast')
add_column_function(map_ecoli, ecoli_ppi, 83333, 'Homologous in Ecoli')

#--------FUNCTION DATABASE SEARCH------#

#import datasets to search in

mint_ds=pd.read_csv(".Datasets/homology files/human_MINT_curated.csv")
dip_ds=pd.read_csv(".Datasets/homology files/human_DIP_curated.csv")
apid_ds=pd.read_csv(".Datasets/homology files/human_APID_curated.csv")
biogrid_ds=pd.read_csv(".Datasets/homology files/human_BIOGRID_curated.csv")

##----main function--------#

def database_search(test_dataset_f, database_dataset, database_name):
    pair_list=[]
    for index, rows in test_dataset_f.iterrows():
        my_list =[rows.uidA, rows.uidB]
        pair_list.append(my_list)

    ppi_list=[]
    for index, rows in database_dataset.iterrows():
        my_list =[rows.uidA, rows.uidB]
        ppi_list.append(my_list)

    value_list=[]
    for i in pair_list:
        for j in ppi_list:
            if set(i)==set(j):
                value_list.append(1)
            else:
                value_list.append(0)

    split_sp_lists = [value_list[x:x+len(ppi_list)] for x in range(0, len(value_list), len(ppi_list))]
    final_list=[]
    for i in split_sp_lists:
        if 1 in i:
            final_list.append(1)
        else:
            final_list.append(0)

    test_dataset_f[database_name]=final_list

database_search(test_ds, mint_ds, 'Exists in MINT?')
database_search(test_ds, dip_ds, 'Exists in DIP?')
database_search(test_ds, apid_ds, 'Exists in APID?')
database_search(test_ds, biogrid_ds, 'Exists in BIOGRID?')

#--------------------------------------------------------------------------#
# export final ds with features
#--------------------------------------------------------------------------#

test_ds.to_csv(path + 'final_output.csv', index=False)