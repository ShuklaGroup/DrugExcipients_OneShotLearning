import pandas as pd
import numpy as np
import os
import glob
import itertools
import random



data = pd.read_csv('Rep_kinases_dataset.csv',sep='\t')
data= data[['CPD_ID','NonstereoAromaticSMILES','Kinase_name']]

#remove duplicates
data=data.drop_duplicates()

df = data[['CPD_ID','Kinase_name']]
kinase= data.Kinase_name.unique()

#find all kinase pairs which don't have overlap inhibitor
df_kinase_comb = pd.DataFrame(columns = ['kinase_X', 'kinase_Y'])
for i in range(0,len(kinase)):    #len(receptors)
    non_overlap_kinase = []
    df_X = df[df['Kinase_name'] == kinase[i]]
  
    df_tmp = df[~df.Kinase_name.isin(list(kinase[range(0,i+1)]))]  
    other_kinase = df_tmp.Kinase_name.unique()

    for j in range(419 - i): 
        df_Y = df_tmp[df_tmp['Kinase_name'] == other_kinase[j]]
        if (sum(df_X.CPD_ID.isin(df_Y.CPD_ID) ) == 0):
            non_overlap_kinase.append(other_kinase[j])
    
    kinase_X = np.repeat(kinase[i], len(non_overlap_kinase))
    d = {'kinase_X':kinase_X,'kinase_Y':non_overlap_kinase}
    df_comb = pd.DataFrame(d)
    df_kinase_comb = df_kinase_comb.append(df_comb)


##write all kinase pairs which don't have overlap inhibitors into csv file (16556 pairs)
df_kinase_comb.to_csv('Non_Overlap_Kinase_pairs.csv',index=False)
Non_overlap_kinaseX = df_kinase_comb.kinase_X.unique()

####for each kinase X, generate negative data using kinaseX + inhibitor from kinaseY 
#####kinaseY is the kinase has no same inhibitors with kinaseX
all_negative_dataset = pd.DataFrame(columns = ['negative_kinase', 'negative_inhibitor'])
for i in range(0,len(Non_overlap_kinaseX)):
    df_comb_sub = df_kinase_comb[df_kinase_comb['kinase_X'] == Non_overlap_kinaseX[i]]
    df_original_sub = df[df.Kinase_name.isin(df_comb_sub.kinase_Y)]
    inhibitor_select = df_original_sub.CPD_ID.unique()
    negative_kinase = np.repeat(Non_overlap_kinaseX[i], len(inhibitor_select))
    d0 = {'negative_kinase':negative_kinase,'negative_inhibitor':inhibitor_select}
    negative_df = pd.DataFrame(d0)
    all_negative_dataset = all_negative_dataset.append(negative_df)

all_negative_dataset.to_csv('all_negative_dataset.csv',index=False)

###random select negative samples
random.seed(7)
for i in range(0,10):
    negative = all_negative_dataset.sample(n = 123005)  ####same number as the positive sample
    negative.to_csv('neg_same_size_data' + str(i) +'.csv',index = False)




