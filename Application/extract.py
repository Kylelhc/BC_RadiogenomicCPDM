
import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor
import logging
logging.getLogger('radiomics').setLevel(logging.ERROR)

def extract(image_np):
  imgNRRD = '/content/drive/MyDrive/Analysis/image/temp.nrrd'
  maskNRRD = '/content/drive/MyDrive/Analysis/mask/temp.nrrd'
  image_sitk = sitk.GetImageFromArray(image_np)
  sitk.WriteImage(image_sitk, imgNRRD)
  mask_np = np.zeros_like(image_np)
  height, width = image_np.shape
  mask_np[height//8:7*height//8, width//8:7*width//8] = 1
  mask_sitk = sitk.GetImageFromArray(mask_np)
  sitk.WriteImage(mask_sitk, maskNRRD)
  extractor = featureextractor.RadiomicsFeatureExtractor()
  result = extractor.execute(imgNRRD, maskNRRD)
  result = {k: v for k, v in result.items() if isinstance(v, float) or (isinstance(v, np.ndarray) and issubclass(v.dtype.type, np.floating))}
  result = {k: v.item() if isinstance(v, np.ndarray) else v for k, v in result.items()}
  features = []
  features.append(result)

  df = pd.DataFrame.from_records(features)
  df = df.filter(regex='original_')
  vector = df.values[0]
  mean = np.mean(vector)
  std = np.std(vector)
  vectorn = (vector - mean) / std

  return vector, vectorn


import os
import pandas as pd
import numpy as np
import torch
import pandas as pd
import os
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from keras.applications.inception_v3 import InceptionV3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.applications.vgg16 import preprocess_input
import numpy as np
from collections import Counter

import sys

clinicalDataPath = sys.argv[1]
imgFolderPath = sys.argv[2]
data_path = sys.argv[3]

clinical = pd.read_csv(clinicalDataPath)
clinicalIDs = clinical['ID'].tolist()

avaID = []
for each in clinicalIDs:
  if each in os.listdir(imgFolderPath):
    avaID.append(each)

images = []
for id in avaID:
  filePath = imgFolderPath + '/' + id + '/' + '000.npy'
  images.append(np.load(filePath).astype(np.float32))

# extract features by using pyrediomic
features_pyredio = []
for image in images:
  vector, vectorn = extract(image[0])
  features_pyredio.append(vector)

# extract features by using VGG16
features_VGG16 = []
model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for img_data in images:
  img_data = np.stack((img_data[0],)*3, axis=-1)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  vgg16_features = model.predict(img_data).flatten()
  scaler = StandardScaler()
  vgg16_features = np.ravel(scaler.fit_transform(vgg16_features.reshape(-1, 1)))
  features_VGG16.append(vgg16_features)
pca = PCA(n_components=0.95)
features_VGG16 = pca.fit_transform(features_VGG16)

# extract features by using ResNet50
features_ResNet50 = []
model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for img_data in images:
  img_data = np.stack((img_data[0],)*3, axis=-1)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  ResNet50_features = model.predict(img_data).flatten()
  scaler = StandardScaler()
  ResNet50_features = np.ravel(scaler.fit_transform(ResNet50_features.reshape(-1, 1)))
  features_ResNet50.append(ResNet50_features)
pca = PCA(n_components=0.95)
features_ResNet50 = pca.fit_transform(features_ResNet50)

# extract features by using ResNet50
features_InceptionV3 = []
model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
for img_data in images:
  img_data = np.stack((img_data[0],)*3, axis=-1)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)
  InceptionV3_features = model.predict(img_data).flatten()
  scaler = StandardScaler()
  InceptionV3_features = np.ravel(scaler.fit_transform(InceptionV3_features.reshape(-1, 1)))
  features_InceptionV3.append(InceptionV3_features)
pca = PCA(n_components=0.95)
features_InceptionV3 = pca.fit_transform(features_InceptionV3)

torch.save({
    'features_pyredio' : features_pyredio,
    'features_VGG16' : features_VGG16,
    'features_ResNet50' : features_ResNet50,
    'features_InceptionV3': features_InceptionV3
  }, data_path)
