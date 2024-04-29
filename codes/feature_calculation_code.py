import pandas as pd
import functools
from pygosemsim import term_set
from pygosemsim import annotation
from pygosemsim import download
from pygosemsim import similarity
from pygosemsim import graph
import networkx as nx
from goatools import obo_parser
import wget
import os
import requests as r
from Bio import SeqIO
from io import StringIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
from urllib import request
import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import operator

#-------------  INPUT DATASET -----------#
test_dataset=pd.read_csv("input_dataset.csv")

# input dataset must contain at least 4 columns named:
# uidA
# uidB
# protein_accession_A
# protein_accession_B



def mega_function_add_features (test1):

    #----------------- (1) GO SIMILARITY CODE------------------#

    G = graph.from_resource("go-basic")
    similarity.precalc_lower_bounds(G)
    annot = annotation.from_resource("goa_human")

    test_list =[]
    for index, rows in test1.iterrows():
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

        test1[similarity_df_header]=go_sim_list

    add_similarity_to_test('biological_process', 'BP_similarity')
    add_similarity_to_test('molecular_function', 'MF_similarity')
    add_similarity_to_test('cellular_component', 'CC_similarity')


    #---------------(2) HOMOLOGY CODE ----------------------#

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

    def function_2(test_dataset_insert):


        test_list =[]
    
        for index, rows in test_dataset_insert.iterrows():
            my_list =[rows.protein_accession_A, rows.protein_accession_B]
            test_list.append(my_list)


        #Create Homologous map list
        #First we find  the HIDs that correspond to each protein in the pair
        # homol_list ---> A list that contains dataframes with 2 rows: The 2 lines from the homologous dataset that contain the protein accession numbers for each protein in the pair
        #Secondly we filter the homologene data to take all the HIDs that match the HIDs from the protein pair 
        # homol_list2 ---> A list that contains dataframes. Each dataframe contains all the HIDs that match each protein in the pair. Eg. each dataset contains the corresponding homolgous NPs for each pair

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

            #we filter each dataframe in homologous dataframe that contains the organism that we want (eg 10090-> we filter the mouse homologous)

            homol_list_filtered=[]
            for i in homol_list2:
                hom1=i.loc[i['taxid']==taxid_no]
                homol_list_filtered.append(hom1)

            #we merge each homologous  dataset with the map dataset so we can match the protein accession numbers with the corresponding uniprot ids

            merge_list=[]
            for ds in homol_list_filtered:
                mrg= pd.merge ( ds,map_ds, left_on='protein_accession', right_on='Protein_Accession')
                merge_list.append(mrg)

            #for each item in merge list:
            # if it has 2 unique HIDs that means it has 2 proteins and means there has been a homologous pair in the other organism. 
            # In that case we create a sublist that contains the corrseponding uniprot ids ad then we append that sublist to 'pair_list'
            #else we append a 'NaN list'

            pair_list=[]
            nan_list=['NaN','NaN']
            for i in merge_list:
                if i['HID'].nunique()==2:
                    sublist=list(i['UniprotID'])
                    pair_list.append(sublist)
                else:
                    pair_list.append(nan_list)

            #we create a PPI list (same as test list) but for each organism that we want to match

            ppi_list=[]
            for index, rows in ppi_ds.iterrows():
                my_list =[rows.uniprotid_A, rows.uniprotid_B]
                ppi_list.append(my_list)
            
            #if there is a matching pair between sblists (pairs) in pair_list and PPI list we append 1
            # if there is NaN in a sublist (that means it is a NAN list- eg there isn;t a homologous pair) of pair list we append NaN

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

            test_dataset_insert[taxid_name ]=final_list

        add_column_function(map_mouse, mouse_ppi, 10090, 'Homologous in Mouse')
        add_column_function(map_drosophila, dros_ppi, 7227, 'Homologous in Drosophila')
        add_column_function(map_yeast, yeast_ppi, 559292, 'Homologous in Yeast')
        add_column_function(map_ecoli, ecoli_ppi, 83333, 'Homologous in Ecoli')


        mint_ds=pd.read_csv(".Datasets/homology files/human_MINT_curated.csv")
        dip_ds=pd.read_csv(".Datasets/homology files/human_DIP_curated.csv")
        apid_ds=pd.read_csv(".Datasets/homology files/human_APID_curated.csv")
        biogrid_ds=pd.read_csv(".Datasets/homology files/human_BIOGRID_curated.csv")


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

        database_search(test_dataset_insert, mint_ds, 'Exists in MINT?')
        database_search(test_dataset_insert, dip_ds, 'Exists in DIP?')
        database_search(test_dataset_insert, apid_ds, 'Exists in APID?')
        database_search(test_dataset_insert, biogrid_ds, 'Exists in BIOGRID?')
        return test_dataset_insert

    function_2(test1)

    #-----------------(3) E VALUE CODE----------------------------#

    test_list =[]
    for index, rows in test1.iterrows():
        my_list =[rows.uidA, rows.uidB]
        test_list.append(my_list)

    seq_list=[]
    nan_list=['NaN','NaN']
    for pair in test_list:
        pair_list=[]

        baseUrl="http://www.uniprot.org/uniprot/"
        def pair_append(string):
            cID_A=string
            currentUrl=baseUrl+cID_A+".fasta"
            response = r.post(currentUrl)
            cData=''.join(response.text)
            Seq1=StringIO(cData)
            pSeq=list(SeqIO.parse(Seq1,'fasta'))
            if len(pSeq)!=0:
                sequence=pSeq[0]
                fasta=sequence.format("fasta")
                fasta_str=str(fasta)
                final_seq=fasta_str.splitlines()
                del final_seq[0]
                final=' '.join([str(elem) for elem in final_seq])
                final=final.replace(" ","")
                pair_list.append(final)
            else:
                pair_list.append('NaN')
        pair_append(pair[0])
        pair_append(pair[1])

        seq_list.append(pair_list)

    aligner = Align.PairwiseAligner(match_score=1.0)
    score_list=[]
    for seq_pair in seq_list:
        if ((seq_pair[0]!='NaN')& (seq_pair[0]!='NaN')):
            score = aligner.score(seq_pair[0], seq_pair[1])
            score_list.append(score)
        else:
            score_list.append('NaN')

    mean_pair_length=[]
    for pair in seq_list:
        if ((pair[0]!='NaN')& (pair[0]!='NaN')):
            mean=(len(pair[0])+len(pair[1]))/2
            mean_pair_length.append(mean)
        else:
            mean_pair_length.append('NaN')

    d={'score':score_list,'meanlen':mean_pair_length}
    df=pd.DataFrame(d, columns=['score' , 'meanlen'])
    calc_list =[]
    for index, rows in df.iterrows():
        my_list =[rows.score, rows.meanlen]
        calc_list.append(my_list)

    e_list=[]
    for i in calc_list:
        if i!=['NaN', 'NaN']:
            e_val=(i[1])*(2.7**(-i[0]))
            e_list.append(e_val)
        else:
            e_list.append('NaN')

    test1['Sequence_similarity']=e_list

    #-----------------(4) INTERPRO CODE----------------------------#

    def add_pfam_sim(test_ds):
        testA_list = test_ds.uidA.values.tolist()
        testB_list = test_ds.uidB.values.tolist()
        
        pfam_ids_testA=[]
        for prot in testA_list:
            i=prot.strip("''")
            url = "https://www.ebi.ac.uk/interpro/api/entry/all/protein/UniProt/"+i+"/"

            req = request.Request(url)
            response = request.urlopen(req)
            encoded_response = response.read()
            decoded_response = encoded_response.decode()
            try:
                payload = json.loads(decoded_response)
            except json.decoder.JSONDecodeError:
                payload='NaN'

            ipr_list=[]
            if payload!='NaN':
                for item in payload['results']:
                    if item['metadata']['source_database']=='pfam':
                        ipr_list.append(item['metadata']['accession'])
                pfam_ids_testA.append(ipr_list)
            else:
                pfam_ids_testA.append('NaN')
        pfam_A=[]
        for i in pfam_ids_testA:
            if len(i)==0:
                pfam_A.append('NaN')
            else:
                pfam_A.append(i)

        pfam_ids_testB=[]
        for prot in testB_list:
            i=prot.strip("''")
            url = "https://www.ebi.ac.uk/interpro/api/entry/all/protein/UniProt/"+i+"/"

            req = request.Request(url)
            response = request.urlopen(req)
            encoded_response = response.read()
            decoded_response = encoded_response.decode()
            try:
                payload = json.loads(decoded_response)
            except json.decoder.JSONDecodeError:
                payload='NaN'

            ipr_list=[]
            if payload!='NaN':
                for item in payload['results']:
                    if item['metadata']['source_database']=='pfam':
                        ipr_list.append(item['metadata']['accession'])
                pfam_ids_testB.append(ipr_list)
            else:
                pfam_ids_testB.append('NaN')
        pfam_B=[]
        for i in pfam_ids_testB:
            if len(i)==0:
                pfam_B.append('NaN')
            else:
                pfam_B.append(i)


        pfam_pair_list=[]
        pfam_pair_list.extend([list(x) for x in zip(pfam_A, pfam_B)])

        list_ds=[]
        with open(".Datasets/3did_flat_Mar_4_2021.dat") as file:
            lines = file.readlines()
            for x in lines:
                if x.startswith('#=ID'):
                    list_ds.append(x)


        interact_list=[]
        for i in list_ds:
            i2=i.split("\t")
            p_A=i2[3].strip(' (')
            p_A=p_A.split(".")
            p_A=p_A[0]
            p_B=i2[4].split(".")
            p_B=p_B[0]
            i3=[p_A,p_B]
            interact_list.append(i3)


        result_list=[]
        for i in pfam_pair_list:
            for j in interact_list:
                if ((j[0] in i[0])& (j[1] in i[1])):
                    result_list.append(1)
                elif((j[0] in i[1])& (j[1] in i[0])):
                    result_list.append(1)
                elif ((i[0]=='NaN')|(i[1]=='NaN')):
                    result_list.append('NaN')
                else:
                    result_list.append(0)

        split_sp_lists = [result_list[x:x+len(interact_list)] for x in range(0, len(result_list), len(interact_list))]
        final_list=[]
        for i in split_sp_lists:
            if 1 in i:
                final_list.append(1)
            elif 'NaN' in i:
                final_list.append('NaN')
            else:
                final_list.append(0)
        test_ds['pfam_interaction']=final_list


    add_pfam_sim(test1)

    #----------------------- (5) SL CODE -----------------------------#
    map_ds=pd.read_csv(".Datasets/SL files/map_uniprot_human.csv", usecols=['ID', 'Protein_Accession'])
    es_ds=pd.read_table(".Datasets/SL files/eSLDB_Homo_sapiens.txt", delimiter='\t', usecols=['Experimental annotation', 'SwissProt entry'])
    es_ds = es_ds[es_ds["Experimental annotation"].str.contains("None") == False]
    es_ds = es_ds[es_ds["SwissProt entry"].str.contains("None") == False]
    es_ds.rename(columns={'Experimental annotation':'Experimental_annotation', 'SwissProt entry':'Protein_Accession'}, inplace=True)
    es_ds_2=es_ds.Experimental_annotation.str.split(',')
    es_ds['Experimental_annotation'] = es_ds_2


    test_list =[]
    for index, rows in test1.iterrows():
        my_list =[rows.uidA, rows.uidB]
        test_list.append(my_list)

    merge_map=[]
    for pair in test_list:
        if pair[0]==pair[1]:      
    # in this code:
    # if two proteins are the same we append 1 from the beggining (they will have the same co-localization)
            merge_map.append('same')
        else:
            match=map_ds[map_ds["ID"].isin(pair) == True]
            merge_map.append(match)


    es_list=[]
    for pair in merge_map:
        if 'same' in pair:
            es_list.append('same')   
        else:
            mrg= pd.merge ( pair,es_ds, on='Protein_Accession')
            es_list.append(mrg)

    es_list_new=[]
    for df in es_list:
        if 'same' in df:
            es_list_new.append('same')
        else:
            df2=df[['ID','Protein_Accession', 'Experimental_annotation']].loc[df[['ID','Protein_Accession', 'Experimental_annotation']].astype(str).drop_duplicates().index]
            es_list_new.append(df2)

    values_list=[]
    for df in es_list_new:
        if 'same' in  df:
            values_list.append(1)
        elif len(df)>=2:
            list1=df.Experimental_annotation.tolist()
            values_list.append(list1)
        else:
            values_list.append('NaN')

    final_list=[]

    for pair in values_list:
        if pair==1:
            final_list.append(1)
        elif pair=='NaN':
            final_list.append('NaN')
        else:
            new_list_A=[]
            new_list_B=[]
            new_list=[new_list_A, new_list_B]
            for s in pair[0]:
                s=s.replace(" ","")
                new_list_A.append(s)
            for s in pair[1]:
                s=s.replace(" ","")
                new_list_B.append(s)
            count=0
            for i in new_list_A:
                for j in new_list_B:
                    if i==j:
                        count=count+1
            if count>0:
                final_list.append(1)
            else:
                final_list.append(0)
            
            
    test1['Subcellular Co-localization?']=final_list

    #--------------------(6) SPEARMAN FUNCTION -------------------#

    map_test=pd.read_csv(".Datasets/spearman files/map.csv")
    map_test.rename(columns={'UniProtKB-AC':'UniprotID'}, inplace=True)
    df1 = pd.read_table(".Datasets/spearman files/GDS181.soft", delimiter="\t", skiprows=[i for i in range(306,)])
    df2 = pd.read_table(".Datasets/spearman files/GDS531.soft", delimiter="\t", skiprows=[i for i in range(210,)])
    df3 = pd.read_table(".Datasets/spearman files/GDS807.soft", delimiter="\t", skiprows=[i for i in range(98,)], comment='!')
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
    for i in GDtotal:
        i = i.fillna(i.iloc[2:].mean())
        i=i.drop(columns=['ID_REF'])
        i=i.groupby('IDENTIFIER').mean().reset_index()
        GD_test.append(i)

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

    test_dataset_final= my_function(GD_test, test1, map_test)

    #-------------(7) EXTRA FEATURES ----------------------------------------#
    def test_list(testds):    
        test_list =[]
        for index, rows in testds.iterrows():
            my_list =[rows.uidA, rows.uidB]
            test_list.append(my_list)

        seq_list=[]
        nan_list=['NaN','NaN']
        for pair in test_list:
            pair_list=[]

            baseUrl="http://www.uniprot.org/uniprot/"
            def pair_append(string):
                cID_A=string
                currentUrl=baseUrl+cID_A+".fasta"
                response = r.post(currentUrl)
                cData=''.join(response.text)
                Seq1=StringIO(cData)
                pSeq=list(SeqIO.parse(Seq1,'fasta'))
                if len(pSeq)!=0:
                    sequence=pSeq[0]
                    fasta=sequence.format("fasta")
                    fasta_str=str(fasta)
                    final_seq=fasta_str.splitlines()
                    del final_seq[0]
                    final=' '.join([str(elem) for elem in final_seq])
                    final=final.replace(" ","")
                    pair_list.append(final)
                else:
                    pair_list.append('NaN')
            pair_append(pair[0])
            pair_append(pair[1])

            seq_list.append(pair_list)
        return (seq_list)
    

    sequences1= test_list(test_dataset_final)


    sequences=[]
    for i in sequences1:
        j=i[0].replace('X','')
        l=i[1].replace('X','')
        list_temp=[j,l]
        sequences.append(list_temp)

    def get_features(X):
        
        aaper=X.get_amino_acids_percent()['A']
        laper=X.get_amino_acids_percent()['L']
        iaper=X.get_amino_acids_percent()['I']
        maper=X.get_amino_acids_percent()['M']
        faper=X.get_amino_acids_percent()['F']
        vaper=X.get_amino_acids_percent()['V']
        saper=X.get_amino_acids_percent()['S']
        paper=X.get_amino_acids_percent()['P']
        taper=X.get_amino_acids_percent()['T']
        yaper=X.get_amino_acids_percent()['Y']
        haper=X.get_amino_acids_percent()['H']
        qaper=X.get_amino_acids_percent()['Q']
        naper=X.get_amino_acids_percent()['N']
        kaper=X.get_amino_acids_percent()['K']
        daper=X.get_amino_acids_percent()['D']
        eaper=X.get_amino_acids_percent()['E']
        caper=X.get_amino_acids_percent()['C']
        waper=X.get_amino_acids_percent()['W']
        raper=X.get_amino_acids_percent()['R']
        gaper=X.get_amino_acids_percent()['G']
        

        mw= X.molecular_weight()

        arom= X.aromaticity()

        insta=X.instability_index()

        fraction=X.secondary_structure_fraction() 
        helix_fraction=fraction[0]
        turn_fraction=fraction[1]
        sheet_fraction=fraction[2]

        extinct= X.molar_extinction_coefficient() 
        cys_reduced=extinct[0]
        cys_residues=extinct[1]

        gravy=X.gravy(scale='KyteDoolitle')

        ph_charge=X.charge_at_pH(7)

        
        listf=[aaper, laper, faper,iaper,maper, vaper, saper, paper,taper,yaper,haper,qaper,naper,kaper,daper,eaper,caper,waper,raper,gaper
            
            , mw, arom, insta, 
            helix_fraction, turn_fraction, sheet_fraction,
            
                cys_reduced,cys_residues,
                gravy,ph_charge ]
        return(listf)

    feature_list=[]
    for seq in sequences:
        X0=ProteinAnalysis(seq[0])
        seq0=get_features(X0)
        X1=ProteinAnalysis(seq[1])
        seq1=get_features(X1)
        list1=[seq0,seq1]
        feature_list.append(list1)

   

    substr_list=[]
    for pair in feature_list:
        s = list(map(operator.sub, pair[0], pair[1]))
        substr_list.append(s)
    

    abs_list=[]
    for i in substr_list:
        abslist=[]
        for l in i:
            new_val=abs(l)
            abslist.append(new_val)
        abs_list.append(abslist)

    

    new_df= pd.DataFrame(abs_list, columns=['A %', 'L %','F %','I %','M %','V %','S %','P %','T %','Y %','H %','Q %', 'N %', 'K %','D %','E %','C %','W %','R %','G %',
                                            'MW dif','Aromaticity dif', 'Instability dif',
                                            'helix_fraction_dif','turn_fraction_dif','sheet_fraction_dif',
                                            'cys_reduced_dif','cys_residues_dif',
                                            'gravy_dif', 'ph7_charge_dif'])

    test_dataset_final=pd.concat([test_dataset_final,new_df], axis=1)

    return (test_dataset_final)

test_dataset_new= mega_function_add_features (test_dataset)


#----------Extra RNA features 1------------------------------#

test_uids=test_dataset_new['uidA', 'uidB']


#     GSE227375_norm_counts_STAR

expression_ds=pd.read_csv(".Datasets/extra rna seq columns/GSE227375_norm_counts_STAR.csv")
expression_ds.columns = expression_ds.columns.str.replace('Unnamed: 0','ENS_ID')

#----------------------------------------------------------------#
# This dataset contains gene entries in ENSEMBLE IDs: ENSG00....
#-----------------------------------------------------------------#
#  Map UIDS with ENSG
#------------------------------------------------------------------#
map_ds=pd.read_csv(".Datasets/extra rna seq columns/UID_to_ENSBL_mapping.csv", usecols=['uid', 'ID'])
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

exp_list =[]
for index, rows in concatds.iterrows():
    my_list =[rows.expression_list_A, rows.expression_list_B]
    exp_list.append(my_list)


calc_list=[]
for pair in exp_list:
    if (str(pair[0])=='nan')|(str(pair[1])=='nan'):
        calc_list.append('NaN')
    else:
        
        rho, p= spearmanr(pair[0], pair[1])
        calc_list.append(rho)



test_dataset_new['GSE227375_spearman'] =calc_list



#----------Extra rna features 2------------------------------#

test_uids=test_dataset_new['uidA', 'uidB']

#     GSE228702_adjusted_expression_Cellcounts_granulatorAbis0_nnls

expression_ds=pd.read_csv(".Datasets/extra rna seq columns/GSE228702_adjusted_expression_Cellcounts_granulatorAbis0_nnls.csv")
expression_ds.columns = expression_ds.columns.str.replace('Unnamed: 0','ENS_ID')


#----------------------------------------------------------------#
# This dataset contains gene entries in GeneCards IDs
#-----------------------------------------------------------------#
#  Map UIDS with GeneCards
#------------------------------------------------------------------#
map_ds=pd.read_csv(".Datasets/extra rna seq columns/UID_to_GeneCards_mapping.csv", usecols=['uid', 'ID'])
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


exp_list =[]
for index, rows in concatds.iterrows():
    my_list =[rows.expression_list_A, rows.expression_list_B]
    exp_list.append(my_list)


calc_list=[]
for pair in exp_list:
    if (str(pair[0])=='nan')|(str(pair[1])=='nan'):
        calc_list.append('NaN')
    else:
        
        rho, p= spearmanr(pair[0], pair[1])
        calc_list.append(rho)


test_dataset_new['GSE228702_spearman'] =calc_list


#--------------------------------------------------------------------------------------------#
# save file
#--------------------------------------------------------------------------------------------#


test_dataset_new.to_csv('feature_calculation_output.csv', index=False)

