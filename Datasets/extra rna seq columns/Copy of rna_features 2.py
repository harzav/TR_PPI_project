import pandas as pd
import numpy as np
from scipy.stats import spearmanr

test_ds= pd.read_csv(r"C:\Users\harry\Desktop\Final_Dataset_curations\final_dataset_extended.csv")
test_uids=pd.read_csv(r"C:\Users\harry\Desktop\Final_Dataset_curations\final_dataset_extended.csv", usecols=['uidA', 'uidB'])
test_uids=test_uids.sample(n=10)
print(test_uids)

#----------import rna dataset 2------------------------------#

#     GSE228702_adjusted_expression_Cellcounts_granulatorAbis0_nnls

expression_ds=pd.read_csv(r"C:\Users\harry\Desktop\GSE228702_adjusted_expression_Cellcounts_granulatorAbis0_nnls.csv")
expression_ds.columns = expression_ds.columns.str.replace('Unnamed: 0','ENS_ID')
#expression_ds['expression_list']= expression_ds.loc[:, expression_ds.columns != 'ENS_ID'].values.tolist()
#expression_ds = expression_ds[['ENS_ID', 'expression_list']]


#----------------------------------------------------------------#
# This dataset contains gene entries in GeneCards IDs
#-----------------------------------------------------------------#
#  Map UIDS with GeneCards
#------------------------------------------------------------------#
map_ds=pd.read_csv(r"C:\Users\harry\UID_to_GeneCards_mapping.csv", usecols=['uid', 'ID'])
map_ds['ID'] = map_ds['ID'].str.split('.').str[0]

test_mapped= pd.merge(test_uids, map_ds, left_on='uidA', right_on='uid', how='left')
test_mapped.rename(columns={'ID':'ID_A'}, inplace=True)
test_mapped= pd.merge(test_mapped, map_ds, left_on='uidB', right_on='uid', how='left')
test_mapped.rename(columns={'ID':'ID_B'}, inplace=True)

#---------------- A-------------------------------------#
map_merge_A= pd.merge(test_mapped, expression_ds, left_on='ID_A', right_on='ENS_ID', how='left')
del map_merge_A['uid_x']
del map_merge_A['uid_y']
#------ calculate average between expressions lists of different RNAs of the same protein
group_map_merge_A=map_merge_A.groupby(['ENS_ID'], sort=False).mean()
group_map_merge_A['expression_list_A']= group_map_merge_A.loc[:, group_map_merge_A.columns != 'ENS_ID'].values.tolist()

map_merge_A=map_merge_A[['uidA', 'uidB','ENS_ID']]
merge_A=pd.merge(map_merge_A, group_map_merge_A,on='ENS_ID', how='left')
merge_A=merge_A[['uidA', 'uidB','expression_list_A']]


#---------------------- B--------------------------#
map_merge_B= pd.merge(test_mapped, expression_ds, left_on='ID_B', right_on='ENS_ID', how='left')
del map_merge_B['uid_x']
del map_merge_B['uid_y']
group_map_merge_B=map_merge_B.groupby(['ENS_ID'], sort=False).mean()
group_map_merge_B['expression_list_B']= group_map_merge_B.loc[:, group_map_merge_B.columns != 'ENS_ID'].values.tolist()

map_merge_B=map_merge_B[['uidA', 'uidB','ENS_ID']]
merge_B=pd.merge(map_merge_B, group_map_merge_B,on='ENS_ID', how='left')
merge_B=merge_B[['uidA', 'uidB','expression_list_B']]


# DROP DUPLICATES AND! NANs
#----------------------------------------------------------#
merge_A = merge_A.drop_duplicates(subset=['uidA','uidB'])
merge_B = merge_B.drop_duplicates(subset=['uidA','uidB'])
merge_B =merge_B.drop(['uidA', 'uidB'], axis=1)
#----------------------------------------------------------#



concatds=pd.concat([merge_A, merge_B], axis=1)
print(concatds)


exp_list =[]
for index, rows in concatds.iterrows():
    my_list =[rows.expression_list_A, rows.expression_list_B]
    exp_list.append(my_list)

#print(exp_list)

calc_list=[]
for pair in exp_list:
    if (str(pair[0])=='nan')|(str(pair[1])=='nan'):
        calc_list.append('NaN')
    else:
        
        rho, p= spearmanr(pair[0], pair[1])
        calc_list.append(rho)

print(calc_list)


test_uids['GSE228702_spearman'] =calc_list

print(test_uids)