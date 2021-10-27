import pandas as pd
import numpy as np
import os
import glob
import itertools

#####read the positive sample
data = pd.read_csv('Rep_kinases_dataset.csv',sep='\t')
data= data[['CPD_ID','NonstereoAromaticSMILES','Kinase_name']]


#remove duplicates
data=data.drop_duplicates()
data_pos = data[['CPD_ID','Kinase_name']]
#####read the generated negative samples
data_neg = pd.read_csv('neg_same_size_data0.csv')
data_neg = data_neg[['negative_inhibitor','negative_kinase']]
data_neg = data_neg.rename(columns = {'negative_inhibitor':'CPD_ID', 'negative_kinase' : 'Kinase_name'})

#####combine the positive and negative samples
data_pos_neg = data_pos.append(data_neg)

kinase = data.Kinase_name.unique()
inhibitor = data.CPD_ID.unique()
df = pd.DataFrame(list(itertools.product(inhibitor,kinase)),columns=['CPD_ID','Kinase_name'])

df_merge_pos = pd.merge(df, data_pos, on = ['CPD_ID','Kinase_name'], how = 'left',indicator = 'pos_exist')
df_merge_pos['pos_exist'] = np.where(df_merge_pos.pos_exist == 'both', 1, 2)
df_merge_pos['pos_exist'].value_counts()
df_merge_pos_neg = pd.merge(df_merge_pos, data_neg, on = ['CPD_ID','Kinase_name'], how = 'left',indicator = 'neg_exist')
df_merge_pos_neg['neg_exist'] = np.where(df_merge_pos_neg.neg_exist == 'both', 0, 2)
df_merge_pos_neg['neg_exist'].value_counts()

df_merge_pos_neg.loc[(df_merge_pos_neg['pos_exist'] == 1) & (df_merge_pos_neg['neg_exist'] == 2), 'indicator'] = 1
df_merge_pos_neg.loc[(df_merge_pos_neg['pos_exist'] == 2) & (df_merge_pos_neg['neg_exist'] == 0), 'indicator'] = 0
df_merge_pos_neg.loc[(df_merge_pos_neg['pos_exist'] == 2) & (df_merge_pos_neg['neg_exist'] == 2), 'indicator'] = ''

df_merge_pos_neg1 = df_merge_pos_neg.drop(columns = ['pos_exist','neg_exist'])
df_merge_pos_neg1['indicator'].value_counts()

df_wide= df_merge_pos_neg1.pivot(index='CPD_ID', columns='Kinase_name', values='indicator')
smile=data[['CPD_ID','NonstereoAromaticSMILES']].drop_duplicates()
df_wide = pd.merge(df_wide,smile, on=['CPD_ID'], how='inner')

#include same size pos and neg, and write the input file
df_wide.to_csv("Input_pos_neg_0.csv",index=False)





