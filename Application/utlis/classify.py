"""

This script uses classical ML methods and deep learning methods to do the multi-class classification

Requirements: scikeras, dill

Before run codes, please first set proper configs

"""

import pandas as pd
import numpy as np
from sklearn import datasets
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, cohen_kappa_score, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, average_precision_score
from scipy.stats import uniform, randint
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import dill as pickle
import os, shap
import warnings
warnings.simplefilter(action='ignore', category=Warning)

class configs():
  """
  set up all parameters in this scrip
  """
  MLGlobalRandomSeed = 53
  # split dataset
  stratify = True
  test_size = 0.2
  random_state = 42
  n_iter = 200
  cv = 10
  # traditional ML
  method = {
    'SVC' : {
      'param' : {
        'estimator__C': uniform(0.1, 10),
        'estimator__gamma': ['scale', 'auto'] + list(uniform(0.1, 10).rvs(1000)),
        'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
      },
      'grid' : {
        'n_iter' : n_iter,
        'cv' : cv
      }
    },
    'LG' : { # logistic regression
      'param' : {
        'C': uniform(0.01, 10),
        'solver': ['newton-cg', 'sag', 'saga', 'lbfgs'],
        'max_iter': randint(100, 10000)
      },
      'grid' : {
        'n_iter' : n_iter,
        'cv' : cv
      }
    },
    'XGBoost' : {
      'param' : { # placeholder
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(10, 200),
        'max_depth': randint(1, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 2),
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.1, 1, 10]
      },
      'grid' : {
        'n_iter' : n_iter,
        'cv' : cv
      }
    },
    'Lasso' : {
      'param' : {
        'C': uniform(0.01, 10),
        'solver': ['saga'],
        'max_iter': randint(100, 10000)
      },
      'grid' : {
        'n_iter' : n_iter,
        'cv' : cv
      }
    },
    'NaiveBayes' : {
      'param' : {
        'var_smoothing': np.logspace(-10, -8, num=1000) 
      },
      'grid' : {
        'n_iter' : n_iter,
        'cv' : cv
      }
    },
    'KNN' : {
      'param' : {
        'n_neighbors': randint(1, 30),
        'weights': ['uniform', 'distance'],
        'p': [1, 2] 
      },
      'grid' : {
        'n_iter' : n_iter,
        'cv' : cv
      }
    },
    'CNN' : {
      'param' : {
        'filters': [32, ],
        'kernel_size': [3, ],
        'dense_units': [64,],
        'optimizer': ['adam', ], # 'rmsprop'],
        'batch_size': [10,],
        'epochs': [150,]
      },
      'grid' : {
        'n_iter' : 1,
        'cv' : 3,
        'random_state' : 42
      }
    }
  }

class classification():

  """
  input: X, y
  output: 
    (1) a csv file storing all performance matrix result
    (2) ROC-AUC curves, PR-AUC curves, confusion matrix
  """

  def __init__(self, X_train, X_test, y_train, y_test, outputDirectory, shapname='') -> None:
    """
    X: A 2D array. Each row is a feature vector. For example, [[1,2,3,4],[5,6,7,8]]
    y: A 1D array. Each element is a class, which may be 0, 1, 2 ... For example, [0,1,2,3]
    """
    self.configs = configs
    # X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.configs.test_size, random_state=self.configs.random_state, stratify=y) if configs.stratify else train_test_split(X, y, test_size=self.configs.test_size, random_state=self.configs.random_state)
    X_train, X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
    sc = StandardScaler()
    self.X_train = sc.fit_transform(X_train)
    self.X_test = sc.transform(X_test)
    self.outputDirectory = outputDirectory
    self.imgDirectory = self.outputDirectory+'/img'
    if not os.path.exists(self.imgDirectory):
      os.makedirs(self.imgDirectory)
    self.shapname = shapname
  
  def search(self, clf, model:str):
    np.random.seed(self.configs.MLGlobalRandomSeed)
    searchCV = RandomizedSearchCV(
      clf, 
      param_distributions=self.configs.method[model]['param'], 
      n_iter=self.configs.method[model]['grid']['n_iter'], 
      cv=self.configs.method[model]['grid']['cv']
    )
    searchCV.fit(self.X_train, self.y_train)
    y_pred_train = searchCV.best_estimator_.predict(self.X_train)
    y_pred_test = searchCV.best_estimator_.predict(self.X_test)
    
    return y_pred_train, y_pred_test

  def SVC(self):
    return self.search(OneVsRestClassifier(SVC()), 'SVC')

  def LG(self):
    return self.search(LogisticRegression(multi_class='multinomial'), 'LG')

  def XGBoost(self):
    return self.search(xgb.XGBClassifier(), 'XGBoost')

  def Lasso(self):
    return self.search(LogisticRegression(penalty='l1'), 'Lasso')
  
  def NaiveBayes(self):
    return self.search(GaussianNB(), 'NaiveBayes')

  def KNN(self):
    return self.search(KNeighborsClassifier(), 'KNN')
    
  def CNN(self):
    encoder = OneHotEncoder(sparse=False)
    y_train_one_hot = encoder.fit_transform(self.y_train.reshape(-1, 1))
    y_test_one_hot = encoder.transform(self.y_test.reshape(-1, 1))
    X_train_cnn = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
    X_test_cnn = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
    
    class CustomKerasClassifier(KerasClassifier):
      def __init__(self, filters=32, kernel_size=3, dense_units=64, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.filters = filters
          self.kernel_size = kernel_size
          self.dense_units = dense_units

      def set_params(self, **params):
          super().set_params(**params)
          self.filters = params.get('filters', self.filters)
          self.kernel_size = params.get('kernel_size', self.kernel_size)
          self.dense_units = params.get('dense_units', self.dense_units)
          return self
          
    def create_model(filters=32, kernel_size=3, dense_units=64, optimizer='adam'):
      num_class = len(np.unique(np.concatenate((self.y_train, self.y_test))))
      model = keras.models.Sequential([
          keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
          keras.layers.MaxPooling1D(pool_size=2),
          keras.layers.Flatten(),
          keras.layers.Dense(dense_units, activation='relu'),
          keras.layers.Dense(num_class, activation='softmax')
      ])
      model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
      return model

    model = CustomKerasClassifier(model=create_model, verbose=0)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=self.configs.method['CNN']['param'], n_iter=self.configs.method['CNN']['grid']['n_iter'], cv=self.configs.method['CNN']['grid']['cv'], verbose=0, n_jobs=1, random_state=self.configs.method['CNN']['grid']['random_state'])
    random_search.fit(X_train_cnn, y_train_one_hot)
    model = random_search.best_estimator_.model_
    predicted_train = np.argmax(model.predict(X_train_cnn, verbose=0), axis=1)
    predicted_test = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
    return predicted_train, predicted_test

  def output(self,filename='results.csv'):
    print('\nTraining...\n') 
    models = ['XGBoost']
    matrices = ['accuracy','macroPrecision','macroRecall','macroF1Score','AUROC','AUPRC','CohensKappa']
    columns=['train_accuracy','test_accuracy','train_precision','test_precision','train_recall','test_recall','train_F1_score','test_F1_score','train_AUROC','test_AUROC','train_AUPRC','test_AUPRC','train_Cohens_Kappa','test_Cohens_Kappa']
    df = pd.DataFrame(index=models, columns=columns).fillna(-1)
    if not os.path.exists(self.outputDirectory):
      os.makedirs(self.outputDirectory)
    NumericalResultsPath = self.outputDirectory + '/' + filename
    for model in models:
      y_pred_train, y_pred_test = getattr(self, model)()
      self.ConfusionMatrix(y_pred_test, model)
      matrix_results = []
      for metrix in matrices:
        train_result, test_result = getattr(self, metrix)(y_pred_train, y_pred_test)
        train_result, test_result = round(train_result, 2), round(test_result, 2)
        matrix_results += [train_result, test_result]
      df.loc[model] = matrix_results
      df.to_csv(NumericalResultsPath, index=True)
      print(model, 'done!')
    # NumericalResultsPath = self.outputDirectory + '/' + 'Performance Matrices.csv'
    # df.to_csv(NumericalResultsPath, index=True)
    print("Confusion Metrices:\n", self.imgDirectory)
    print("\nNumerical results:\n", NumericalResultsPath)
    print('\nDone!')

  def ConfusionMatrix(self, pred_test, modelName):
    cm = confusion_matrix(self.y_test, pred_test)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Reds', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(modelName+' ConfusionMatrix', fontsize=12)
    save_path = self.imgDirectory+'/'+ modelName + '_ConfusionMatrix.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

  def accuracy(self, y_pred_train, y_pred_test):
    train_accuracy = accuracy_score(self.y_train, y_pred_train)
    test_accuracy = accuracy_score(self.y_test, y_pred_test)
    return train_accuracy, test_accuracy

  def macroPrecision(self, y_pred_train, y_pred_test):
    train_macro_precision = precision_score(self.y_train, y_pred_train, average='macro')
    test_macro_precision = precision_score(self.y_test, y_pred_test, average='macro')
    return train_macro_precision, test_macro_precision

  def macroRecall(self, y_pred_train, y_pred_test):
    train_macro_recall = recall_score(self.y_train, y_pred_train, average='macro')
    test_macro_recall = recall_score(self.y_test, y_pred_test, average='macro')
    return train_macro_recall, test_macro_recall

  def macroF1Score(self, y_pred_train, y_pred_test):
    train_macro_f1 = f1_score(self.y_train, y_pred_train, average='macro')
    test_macro_f1 = f1_score(self.y_test, y_pred_test, average='macro')
    return train_macro_f1, test_macro_f1

  def AUROC(self, y_pred_train, y_pred_test):
    real, pred = self.binarize(self.y_train, y_pred_train)
    train_auroc = roc_auc_score(real, pred, average='macro', multi_class='ovo')
    real, pred = self.binarize(self.y_test, y_pred_test)
    test_auroc = roc_auc_score(real, pred, average='macro', multi_class='ovo')
    return train_auroc, test_auroc
  
  def AUPRC(self, y_pred_train, y_pred_test):
    real, pred = self.binarize(self.y_train, y_pred_train)
    train_auprc = average_precision_score(real, pred, average='macro')
    real, pred = self.binarize(self.y_test, y_pred_test)
    test_auprc = average_precision_score(real, pred, average='macro')
    return train_auprc, test_auprc

  def CohensKappa(self, y_pred_train, y_pred_test):
    train_cohen_kappa = cohen_kappa_score(self.y_train, y_pred_train)
    test_cohen_kappa = cohen_kappa_score(self.y_test, y_pred_test)
    return train_cohen_kappa, test_cohen_kappa

  def binarize(self, real, pred):
    n_classes = len(np.unique(np.concatenate((self.y_train, self.y_test))))
    real = label_binarize(real, classes=np.arange(n_classes))
    pred = label_binarize(pred, classes=np.arange(n_classes))
    return real, pred
    
def run_classify(dirname, X, y,filename=''):
  outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
  for train_idx, test_idx in outer_cv.split(X):
      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]
      outputDirectory = os.getcwd() + '/predictions/' + dirname
      classifier = classification(X_train, X_test, y_train, y_test, outputDirectory)
      classifier.output(filename=filename)

def run_classify(dirname,x_train, x_test, y_train, y_test,filename='',shapname=''):
  """
  Usage:
  X : 2D nparray --> feature vectors
  y : 1D nparray --> labels
  """
  print("Classify -", dirname)
  
  outputDirectory = os.getcwd() + '/predictions/' + dirname
  classifier = classification(x_train, x_test, y_train, y_test, outputDirectory, shapname)
  classifier.output(filename=filename)

  # x_trainp = root + name + '/x_train.csv'
  # x_testp = root + name + '/x_test.csv'
  # y_trainp = root + name + '/y_train.csv'
  # y_testp = root + name + '/y_test.csv'

  # x_train = np.genfromtxt(x_trainp, delimiter=',', dtype=np.float64)
  # y_train = np.genfromtxt(y_trainp, delimiter=',', dtype=np.uint8)
  # x_test = np.genfromtxt(x_testp, delimiter=',', dtype=np.float64)
  # y_test = np.genfromtxt(y_testp, delimiter=',', dtype=np.uint8)

  
  # # go to your directory to see results.





