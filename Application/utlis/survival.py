
import pandas as pd
import warnings
import os
import torch
import numpy as np
import random
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    'penalizer': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],  
    'tie_method': ['efron', 'breslow', 'exact'],
    'step_size': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
}

kf = KFold(n_splits=10, shuffle=True)

# Function to evaluate the model
def evaluate_model(params, data, kf):
    penalizer = params['penalizer']
    alpha = params['alpha']
    tie_method = params['tie_method']
    step_size = params['step_size']

    train_concordance_indexes = []
    test_concordance_indexes = []
    train_logrank_p_values = []
    test_logrank_p_values = []

    for train_index, test_index in kf.split(data):
        train, test = data.iloc[train_index], data.iloc[test_index]
        
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=alpha, tie_method=tie_method, step_size=step_size)
        cph.fit(train, duration_col='survival_days', event_col='outcome')

        train_c_index = cph.score(train, scoring_method="concordance_index")
        train_concordance_indexes.append(train_c_index)

        test_c_index = cph.score(test, scoring_method="concordance_index")
        test_concordance_indexes.append(test_c_index)

        train['risk_score'] = cph.predict_partial_hazard(train)
        train_median_risk_score = train['risk_score'].median()
        train_group1 = train[train['risk_score'] <= train_median_risk_score]
        train_group2 = train[train['risk_score'] > train_median_risk_score]

        test['risk_score'] = cph.predict_partial_hazard(test)
        test_median_risk_score = test['risk_score'].median()
        test_group1 = test[test['risk_score'] <= test_median_risk_score]
        test_group2 = test[test['risk_score'] > test_median_risk_score]

        train_logrank_results = logrank_test(
            durations_A=train_group1['survival_days'],
            durations_B=train_group2['survival_days'],
            event_observed_A=train_group1['outcome'],
            event_observed_B=train_group2['outcome']
        )
        train_logrank_p_values.append(train_logrank_results.p_value)

        test_logrank_results = logrank_test(
            durations_A=test_group1['survival_days'],
            durations_B=test_group2['survival_days'],
            event_observed_A=test_group1['outcome'],
            event_observed_B=test_group2['outcome']
        )
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

for penalizer in param_grid['penalizer']:
    for alpha in param_grid['alpha']:
        for tie_method in param_grid['tie_method']:
            for step_size in param_grid['step_size']:
                params = {'penalizer': penalizer, 'alpha': alpha, 'tie_method': tie_method, 'step_size': step_size}
                results = evaluate_model(params, combined_data, kf)
                print(f"Params: {params}, Results: {results}")

                if results['mean_test_c_index'] > best_score:
                    best_score = results['mean_test_c_index']
                    best_params = params
                    best_results = results

print(f"Best Params: {best_params}, Best Results: {best_results}")

# Fit the final model with the best hyperparameters
cph = CoxPHFitter(
    penalizer=best_params['penalizer'], 
    l1_ratio=best_params['alpha'], 
    tie_method=best_params['tie_method'], 
    step_size=best_params['step_size']
)
cph.fit(combined_data, duration_col='survival_days', event_col='outcome')

# Save the model
cph.save('best_coxph_model.pkl')

mean_train_concordance_index = best_results['mean_train_c_index']
mean_test_concordance_index = best_results['mean_test_c_index']
mean_train_logrank_p_value = best_results['mean_train_logrank_p_value']
mean_test_logrank_p_value = best_results['mean_test_logrank_p_value']

print(f"Mean Train Concordance Index: {mean_train_concordance_index}")
print(f"Mean Test Concordance Index: {mean_test_concordance_index}")
print(f"Mean Train Log-Rank P-Value: {mean_train_logrank_p_value}")
print(f"Mean Test Log-Rank P-Value: {mean_test_logrank_p_value}")
