# Pyradiomics software to extract radiomic features and then
# use random forest, svm, XGBoost

from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from scipy.stats import randint
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import xgboost as xgb
import torch, os
import pandas as pd

def calRocPr(best_estimator,x_train,x_test,y_train,y_test):
  train_y_proba = best_estimator.predict_proba(x_train)[:, 1]
  train_roc_auc = roc_auc_score(y_train, train_y_proba)
  print('Train-ROC-AUC:',train_roc_auc)
  test_y_proba = best_estimator.predict_proba(x_test)[:, 1]
  test_roc_auc = roc_auc_score(y_test, test_y_proba)
  print('Test-ROC-AUC:',test_roc_auc)
  precision, recall, thresholds = precision_recall_curve(y_train, train_y_proba)
  train_average_precision = auc(recall, precision)
  print('Train-Precision-Recall:',train_average_precision)
  precision, recall, thresholds = precision_recall_curve(y_test, test_y_proba)
  test_average_precision = auc(recall, precision)
  print('Test-Precision-Recall:',test_average_precision)
  # -------------------------------------------------------- #
  from sklearn.metrics import f1_score
  print('Train F1:')
  cutoff = []
  train_F1 = []
  for threshold in np.arange(0, 1, 0.05):
    y_pred = (train_y_proba > threshold).astype(int)
    f1 = f1_score(y_train, y_pred)
    cutoff.append(threshold)
    train_F1.append(f1)

  ma = max(train_F1)
  cu = cutoff[train_F1.index(ma)]
  print('cutoff',cu,' ','F1',ma)

  print('Test F1:')
  test_F1 = []
  for threshold in np.arange(0, 1, 0.05):
    y_pred = (test_y_proba > threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    test_F1.append(f1)

  ma = max(test_F1)
  cu = cutoff[test_F1.index(ma)]
  print('cutoff',cu,' ','F1',ma)
  return train_y_proba,train_roc_auc,test_y_proba,test_roc_auc,train_average_precision,test_average_precision


def rocPrPlot(y_train, train_y_proba,y_test, test_y_proba,train_roc_auc,test_roc_auc,train_average_precision,test_average_precision):
  fpr, tpr, thresholds = roc_curve(y_train, train_y_proba)
  plt.plot(fpr, tpr, color='#28A745', linestyle='-', label='Train (AUC = {0:.2f})'.format(train_roc_auc), lw=2)
  fpr, tpr, thresholds = roc_curve(y_test, test_y_proba)
  plt.plot(fpr, tpr, color='#FF6B6B', linestyle='-', label='Test (AUC = {0:.2f})'.format(test_roc_auc), lw=2)
  plt.plot([0, 1], [0, 1], color='#A9A9A9', linestyle='-')
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  # plt.title(model_name+' - '+ gene_id +' ROC Curve')
  plt.legend(loc="lower right")
  plt.grid(True)
  plt.show()
  precision, recall, thresholds = precision_recall_curve(y_train, train_y_proba)
  plt.plot(recall, precision, color='#28A745', linestyle='-', label='Train (AUC = {0:.2f})'.format(train_average_precision), lw=2)
  precision, recall, thresholds = precision_recall_curve(y_test, test_y_proba)
  plt.plot(recall, precision, color='#FF6B6B', linestyle='-', label='Test (AUC = {0:.2f})'.format(test_average_precision), lw=2)
  plt.plot([0, 1], [0, 1], color='#A9A9A9', linestyle='-')
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  # plt.title(model_name+' - '+ gene_id +' Precision-Recall Curve')
  plt.legend(loc="lower right")
  plt.grid(True)
  plt.show()

def Logistic(x_train, y_train, x_test, y_test, img=False):
  print('Logistic')

  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import RandomizedSearchCV
  from scipy.stats import uniform

  logistic = LogisticRegression(solver='saga', tol=0.01)
  param_grid = {
      'C': np.logspace(-4, 4, 50),
      'penalty': ['l1', 'l2'],
      'solver': ['liblinear', 'saga']
  }
  grid_search = RandomizedSearchCV(logistic,param_distributions=param_grid,cv=5,scoring='roc_auc', n_jobs=-1)
  grid_search.fit(x_train, y_train)
  print("Best hyperparameters: ", grid_search.best_params_)

  best_estimator = grid_search.best_estimator_
  train_y_proba,train_roc_auc,test_y_proba,test_roc_auc,train_average_precision,test_average_precision=calRocPr(best_estimator,x_train,x_test,y_train,y_test)
  if img: rocPrPlot(y_train, train_y_proba,y_test, test_y_proba,train_roc_auc,test_roc_auc,train_average_precision,test_average_precision)

  return train_roc_auc, test_roc_auc, train_average_precision, test_average_precision


def Lasso(x_train, y_train, x_test, y_test, img=False):
  print('Lasso')

  lasso = LassoCV()
  param_grid = {'alphas': [[0.0001], [0.001], [0.01], [0.1], [1], [10], [100]],
                'cv': [3, 5, 7],
                'max_iter': [100, 500, 1000, 5000],
                'tol': [1e-4, 1e-3, 1e-2],
                }
  grid_search = RandomizedSearchCV(lasso, param_distributions=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
  grid_search.fit(x_train, y_train)
  print("Best hyperparameters: ", grid_search.best_params_)

  best_estimator = grid_search.best_estimator_

  train_y_proba = best_estimator.predict(x_train)
  train_roc_auc = roc_auc_score(y_train, train_y_proba)
  print('Train-ROC-AUC:',train_roc_auc)
  test_y_proba = best_estimator.predict(x_test)
  test_roc_auc = roc_auc_score(y_test, test_y_proba)
  print('Test-ROC-AUC:',test_roc_auc)
  precision, recall, thresholds = precision_recall_curve(y_train, train_y_proba)
  train_average_precision = auc(recall, precision)
  print('Train-Precision-Recall:',train_average_precision)
  precision, recall, thresholds = precision_recall_curve(y_test, test_y_proba)
  test_average_precision = auc(recall, precision)
  print('Test-Precision-Recall:',test_average_precision)

  if img: rocPrPlot(y_train, train_y_proba,y_test, test_y_proba,train_roc_auc,test_roc_auc,train_average_precision,test_average_precision)

  return train_roc_auc, test_roc_auc, train_average_precision, test_average_precision


def RandomForest(x_train, y_train, x_test, y_test, img=False):
  print('Random Forest')

  rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, oob_score=True, bootstrap=True)
  param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 50)],
                'max_depth': [5, 10, 20, 30],
                'min_samples_split': [2, 6, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2'],
                'max_samples': [0.5, 1.0],
                'bootstrap': [True]
                }
  grid_search = RandomizedSearchCV(rf, param_distributions=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
  grid_search.fit(x_train, y_train)
  print("Best hyperparameters: ", grid_search.best_params_)

  best_estimator = grid_search.best_estimator_
  train_y_proba,train_roc_auc,test_y_proba,test_roc_auc,train_average_precision,test_average_precision=calRocPr(best_estimator,x_train,x_test,y_train,y_test)
  if img: rocPrPlot(y_train, train_y_proba,y_test, test_y_proba,train_roc_auc,test_roc_auc,train_average_precision,test_average_precision)

  return train_roc_auc, test_roc_auc, train_average_precision, test_average_precision


def SVM(x_train, y_train, x_test, y_test, img=False):
  print('SVM')

  svc = SVC(random_state=42,probability=True)
  param_grid = {'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'degree': [2, 3, 4],
                'coef0': [0, 1, 2]
                }
  grid_search = RandomizedSearchCV(svc, param_distributions=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
  grid_search.fit(x_train, y_train)
  print("Best hyperparameters: ", grid_search.best_params_)

  best_estimator = grid_search.best_estimator_
  train_y_proba,train_roc_auc,test_y_proba,test_roc_auc,train_average_precision,test_average_precision=calRocPr(best_estimator,x_train,x_test,y_train,y_test)
  if img: rocPrPlot(y_train, train_y_proba,y_test, test_y_proba,train_roc_auc,test_roc_auc,train_average_precision,test_average_precision)

  return train_roc_auc, test_roc_auc, train_average_precision, test_average_precision

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

def XGB(x_train, y_train, x_test, y_test, img=True, para=None):
  print('XGBoost')

  xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
  if para == None:
    param_grid = {
        'n_estimators': [100, 200, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'colsample_bytree': [0.5, 0.7, 0.9, 1],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        'gamma': [0, 0.25, 0.5, 1.0],
        'min_child_weight': [1, 3, 5]
    }
  else:
   param_grid = para

  grid_search = RandomizedSearchCV(xgb, param_distributions=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
  grid_search.fit(x_train, y_train)
  print("Best hyperparameters: ", grid_search.best_params_)

  best_estimator = grid_search.best_estimator_

  train_y_proba,train_roc_auc,test_y_proba,test_roc_auc,train_average_precision,test_average_precision=calRocPr(best_estimator,x_train,x_test,y_train,y_test)
  if img: rocPrPlot(y_train, train_y_proba,y_test, test_y_proba,train_roc_auc,test_roc_auc,train_average_precision,test_average_precision)

  return train_roc_auc, test_roc_auc, train_average_precision, test_average_precision, best_estimator



import sys
clinicalDataPath = sys.argv[1]
imgFolderPath = sys.argv[2]
data_path = sys.argv[3]
flag = sys.argv[4]
fmethod = sys.argv[5]

feature_data = torch.load(data_path)
features = feature_data[fmethod]
combined_data = pd.DataFrame(features,columns=list(range(len(features[0]))))
clinical = pd.read_csv(clinicalDataPath)

clinicalIDs = clinical['ID'].tolist()
avaID = []
for each in clinicalIDs:
  if each in os.listdir(imgFolderPath):
    avaID.append(each)

generatedImgClinical = pd.DataFrame(columns = ['ID', 'er', 'pr', 'her2'])
for each in avaID:
  temp = clinical.loc[clinical['ID']==each]
  generatedImgClinical = generatedImgClinical.append(temp, ignore_index=True)

# print(generatedImgClinical)
combined_data[flag] = generatedImgClinical[flag]

from sklearn.model_selection import train_test_split
train, test = train_test_split(combined_data, test_size=0.1)
y_train = train[flag].tolist()
y_test = test[flag].tolist()
x_train = train.drop([flag], axis=1).values
x_test = test.drop([flag], axis=1).values
xg1,xg2,xg3,xg4,para = XGB(x_train, y_train, x_test, y_test)
print('train_roc_auc:',xg1)
print('test_roc_auc:',xg2)
print('train_pr_auc:',xg3)
print('test_pr_auc:',xg4)
