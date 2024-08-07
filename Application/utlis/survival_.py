import numpy as np
import pandas as pd
import os
import warnings
import torch
import torchtuples as tt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# File paths
tcgaPath = ''  
surDataPath = ''  
imgFolderPath = '' 
data_path = '' 

# Load and preprocess TCGA data
tcga_data = pd.read_csv(tcgaPath, names=['ID', 'Subtype'])
tcga_data['Subtype'] = tcga_data['Subtype'].replace({'Subgroup1': 1, 'Subgroup2': 0})

# Load and preprocess survival data
survival_data = pd.read_csv(surDataPath)
survival_data = survival_data.rename(columns={'Unnamed: 0': 'ID', 'OS_STATUS': 'status', 'OS_MONTHS': 'time'})

# Filter survival data to include only relevant IDs
cleared_survival = pd.DataFrame(columns=['ID', 'time', 'status'])
for each in tcga_data['ID']:
    if each in os.listdir(imgFolderPath):
        temp = survival_data.loc[survival_data['ID'] == each]
        cleared_survival = pd.concat([cleared_survival, temp], ignore_index=True)
survival_data = cleared_survival.astype({'status': 'int'})

# Load feature data
feature_data = torch.load(data_path)
method = 'features_ResNet50'
features = feature_data[method]
survival_days = survival_data['time']
outcomes = survival_data['status']

# Combine data into a single DataFrame
combined_data = pd.DataFrame(features, columns=list(range(len(features[0]))))
combined_data['survival_days'] = survival_days
combined_data['outcome'] = outcomes
combined_data = combined_data.fillna(0)

# Define the parameter grid
param_grid = {
    'num_nodes': [[32, 32], [64, 64], [128, 128]],
    'dropout': [0.1, 0.3, 0.5],
    'batch_norm': [True, False],
    'lr': [0.01, 0.001, 0.0001]
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Function to evaluate the model
def evaluate_model(params, data, kf):
    num_nodes = params['num_nodes']
    dropout = params['dropout']
    batch_norm = params['batch_norm']
    lr = params['lr']

    train_concordance_indexes = []
    test_concordance_indexes = []
    train_logrank_p_values = []
    test_logrank_p_values = []

    cols_standardize = list(range(data.shape[1] - 2))  # Exclude survival_days and outcome
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    x_mapper = DataFrameMapper(standardize)
    
    x_data = x_mapper.fit_transform(data).astype('float32')
    y_data = data[['survival_days', 'outcome']].values

    for train_index, test_index in kf.split(data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        get_target = lambda y: (y[:, 0], y[:, 1])
        y_train = get_target(y_train)
        y_test = get_target(y_test)

        in_features = x_train.shape[1]
        out_features = 1

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        model = CoxPH(net, tt.optim.Adam(lr=lr))

        batch_size = 256
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]

        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose=False, val_data=(x_test, y_test))

        train_c_index = EvalSurv(model.predict_surv_df(x_train), y_train[0], y_train[1], censor_surv='km').concordance_td()
        test_c_index = EvalSurv(model.predict_surv_df(x_test), y_test[0], y_test[1], censor_surv='km').concordance_td()

        train_concordance_indexes.append(train_c_index)
        test_concordance_indexes.append(test_c_index)

        surv_train = model.predict_surv_df(x_train)
        risk_train = -np.log(surv_train).sum(axis=0)
        median_risk_train = np.median(risk_train)
        train_group1 = y_train[0][risk_train <= median_risk_train]
        train_group2 = y_train[0][risk_train > median_risk_train]

        surv_test = model.predict_surv_df(x_test)
        risk_test = -np.log(surv_test).sum(axis=0)
        median_risk_test = np.median(risk_test)
        test_group1 = y_test[0][risk_test <= median_risk_test]
        test_group2 = y_test[0][risk_test > median_risk_test]

        train_logrank_results = logrank_test(train_group1, train_group2, event_observed_A=y_train[1][risk_train <= median_risk_train], event_observed_B=y_train[1][risk_train > median_risk_train])
        train_logrank_p_values.append(train_logrank_results.p_value)

        test_logrank_results = logrank_test(test_group1, test_group2, event_observed_A=y_test[1][risk_test <= median_risk_test], event_observed_B=y_test[1][risk_test > median_risk_test])
        test_logrank_p_values.append(test_logrank_results.p_value)

    mean_test_c_index = np.mean(test_concordance_indexes)
    mean_train_c_index = np.mean(train_concordance_indexes)
    mean_train_logrank_p_value = np.mean(train_logrank_p_values)
    mean_test_logrank_p_value = np.mean(test_logrank_p_values)

    return {
        'mean_test_c_index': mean_test_c_index,
        'mean_train_c_index': mean_train_c_index,
        'mean_train_logrank_p_value': mean_train_logrank_p_value,
        'mean_test_logrank_p_value': mean_test_logrank_p_value
    }

# Grid search
best_score = -np.inf
best_params = None
best_results = None

for num_nodes in param_grid['num_nodes']:
    for dropout in param_grid['dropout']:
        for batch_norm in param_grid['batch_norm']:
            for lr in param_grid['lr']:
                params = {'num_nodes': num_nodes, 'dropout': dropout, 'batch_norm': batch_norm, 'lr': lr}
                results = evaluate_model(params, combined_data, kf)
                print(f"Params: {params}, Results: {results}")

                if results['mean_test_c_index'] > best_score:
                    best_score = results['mean_test_c_index']
                    best_params = params
                    best_results = results

print(f"Best Params: {best_params}, Best Results: {best_results}")




