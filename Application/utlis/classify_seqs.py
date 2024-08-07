import random, torch, os
import numpy as np
import pandas as pd
from classify import run_classify
from sklearn.model_selection import train_test_split

def find_file(filename, search_path):
  for root, dirs, files in os.walk(search_path):
    if filename in files:
      return os.path.join(root, filename)
  return None
  
feat_len = -1  # sequence length
mutation_file_name = ''
label_file_name = ''
clinicals = []  # clinical label names

for cli in clinicals:
    mutation_status_path = find_file(label_file_name, os.getcwd())
    mutations = pd.read_csv(mutation_status_path)

    multi_omic_data_path = find_file(mutation_file_name, os.getcwd())
    multiomic = pd.read_csv(multi_omic_data_path)

    try:
        label_ids = mutations['ID'].tolist()
        omics_ids = multiomic['ID'].tolist()
    except:
        print('No ID column')

    corresponding_labels = []
    for each in omics_ids:
        try:
        label = mutations.loc[mutations['ID'] == each, cli].iloc[0]
        except:
        label = -1
        corresponding_labels.append(label)

    multiomic['label'] = corresponding_labels
    multiomic = multiomic.loc[multiomic['label'] != -1] # fileter

    X = multiomic[multiomic.columns.tolist()[1:feat_len+1]] 
    y = multiomic['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    
    result_path = ''
    dirName = ''
    run_classify(dirName, X_train, X_test, y_train, y_test, result_path)
