from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from sksurv.metrics import integrated_brier_score
import torch
import pandas as pd
import warnings, os,sys
warnings.filterwarnings("ignore")


tcgaPath = sys.argv[1]
surDataPath = sys.argv[2]
imgFolderPath = sys.argv[3]
data_path = sys.argv[4]
method = sys.argv[5]

tcga_data = pd.read_csv(tcgaPath, names=['ID','Subtype'])
tcga_data['Subtype'] = tcga_data['Subtype'].replace({'Subgroup1': 1, 'Subgroup2': 0})
survival_data = pd.read_csv(surDataPath)
survival_data = survival_data.rename(columns={'Unnamed: 0':'ID','OS_STATUS':'status','OS_MONTHS':'time'})
cleared_survival = pd.DataFrame(columns = ['ID', 'time', 'status'])
for each in tcga_data['ID']:
  if each in os.listdir(imgFolderPath):
    temp = survival_data.loc[survival_data['ID']==each]
    cleared_survival = cleared_survival.append(temp, ignore_index=True)

survival_data = cleared_survival.astype({'status':'int'})

feature_data = torch.load(data_path)

features = feature_data[method]
survival_days = survival_data['time']
outcomes = survival_data['status']
combined_data = pd.DataFrame(features,columns=list(range(len(features[0]))))
combined_data['survival_days'] = survival_days
combined_data['outcome'] = outcomes
combined_data = combined_data.fillna(0)

train, test = train_test_split(combined_data, test_size=0.2)
cph = CoxPHFitter(penalizer=0.1)
cph.fit(train, duration_col='survival_days', event_col='outcome')
trainResult = cph.score(train, scoring_method="concordance_index")
testResult = cph.score(test, scoring_method="concordance_index")

print(f"Train C-index: {trainResult}")
print(f"Test C-index: {testResult}")