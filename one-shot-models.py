from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import pandas as pd
import numpy as np
import tensorflow as tf
import deepchem as dc

#load dataset
dataset_file = "Kinase_Input.csv"
df = pd.read_csv("Kinase_Input.csv")
df1 = df.drop(['CPD_ID', 'NonstereoAromaticSMILES'], axis = 1)
Kinase_tasks=list(df1.columns.values)
loader = dc.data.CSVLoader(
      tasks=Kinase_tasks, smiles_field="NonstereoAromaticSMILES", featurizer=dc.feat.ConvMolFeaturizer())
dataset = loader.featurize(dataset_file)
transformers = [dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]
for transformer in transformers:
    dataset = transformer.transform(dataset)



# Number of folds for split 
K = 5
# Depth of attention module
max_depth = 3
# num positive/negative ligands
n_pos = 1
n_neg = 1
# Set batch sizes for network
test_batch_size = 128
support_batch_size = n_pos + n_neg
nb_epochs = 1
n_train_trials = 2000
n_eval_trials = 20
learning_rate = 1e-4
log_every_n_samples = 5000
n_feat = 75


# Define metric
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")
task_splitter = dc.splits.TaskSplitter()
fold_datasets = task_splitter.k_fold_split(dataset, K)



train_folds = fold_datasets[:-1]
train_dataset = dc.splits.merge_fold_datasets(train_folds)
test_dataset = fold_datasets[-1]

support_model = SequentialSupportGraph(n_feat)
support_model.add(GraphConv(64))
support_model.add(GraphPool())
support_model.add(GraphConv(128))
support_model.add(GraphPool())
support_model.add(GraphConv(64))
support_model.add(GraphPool())
support_model.add(tf.keras.layers.Dense(64, tf.nn.tanh))

support_model.add_test(GraphGather(test_batch_size, tf.nn.tanh))
support_model.add_support(GraphGather(support_batch_size, tf.nn.tanh))

#####For AttLSTM, add this layer
#support_model.join(AttnLSTMEmbedding(test_batch_size, support_batch_size, 128,max_depth))

#####For IterRefLSTM, add this layer
#support_model.join(ResiLSTMEmbedding(test_batch_size, support_batch_size, 128,max_depth))


model = SupportGraphClassifier(
    support_model,
    test_batch_size=test_batch_size,
    support_batch_size=support_batch_size,
    learning_rate=learning_rate)

model.fit(
    train_dataset,
    nb_epochs=nb_epochs,
    n_episodes_per_epoch=n_train_trials,
    n_pos=n_pos,
    n_neg=n_neg,
    log_every_n_samples=log_every_n_samples)


mean_scores, std_scores = model.evaluate(
    test_dataset, metric, n_pos, n_neg, n_trials=n_eval_trials)


print("Mean Scores on evaluation dataset")
print(mean_scores)
print("Standard Deviations on evaluation dataset")
print(std_scores)

mean_ = np.array(mean_scores) 
std_ = np.array(std_scores) 
arrays = np.vstack((mean_,std_))
np.savetxt('Res-Siamese-Pos1Neg1.csv', arrays, fmt='%5s', delimiter=',')

