# Conditional Probabilistic Diffusion Model Driven Synthetic Radiogenomic Applications in Breast Cancer

# Introduction
This repository provides the source codes and raw datasets associated with the paper Conditional Probabilistic Diffusion Model (CPDM) Driven Synthetic Radiogenomic Applications in Breast Cancer (BC).

BC exhibits significant heterogeneity, making it crucial to analyze its phenotypic diversity at a multi-omics level for early detection and personalized treatment. The integration of medical images and genomics offers a novel perspective on the study of BC heterogeneity. However, the absence of paired medical images and genomics data could pose a significant challenge.

We proposed the utilization of a well-trained CPDM to address the unpaired data issue in radiogenomic study of BC. The generated images will then be used to predict clinical attributes including gene mutations, estrogen receptor (ER) status, human epidermal growth factor receptor 2-positive (ER+/HER2+) status and have survival significance. The overall project workflow is depicted below.

![workflow](https://github.com/Kylelhc/BC_RadiogenomicCPDM/assets/143105097/39ce3ab5-733e-42bd-be3e-efeb22ce97a6)

# CPDM
## Architecture

![Fig  1](https://github.com/Kylelhc/BC_RadiogenomicCPDM/assets/143105097/922c1fb2-32fb-4f89-91bd-b8e0d75356dc)

## Data

The CPDM is trained with matched patient MRI projections and patient multi-omic data. We repeat the experiment on the gene expression dataset. All datasets are stored in the ```data``` directories. 

- Raw multi-omics data and gene expression data are obtained from the TCGA
- MRIs obtained from the TCIA, where the digital image pixel values are extracted

## Training

To train a CPDM on the training set, using:
```bash
python <VersioinName>.py <mode>
```
- ```<VersioinName>``` represents the CPDM trained on the different dataset, which can be ```multiOmicVersion``` and ```geneExprVersion```
- if user only want to train a CPDM and save the model, please set ```<mode>``` to ```train```
- Please set correct parameters in the , including:
  - ```SideImgPath```: the directory store real MRI projections, i.e. ```../../data/SidePNG128```
  - ```SideBTFpath```: the file store patient ID, i.e. ```../../data/SideViewBTF.csv```. The corresponding patients have paired real MRI projections and genomic profiles.
  - ```allBTFpath```: the file store multi-omic data, i.e. ```../../data/BTF_features.csv```
  - ```geneExprPath```: the file store gene expression data, i.e. ```../../data/TCGABRCA_15gxp.csv```
  - ```timesteps```: 1500
  - ```loss_type```: regularization methods, including ```l1```, ```l2```, and ```l1_l2```
  - ```lr```: learning rate
  - ```batch_size```: 6
  - ```epochs```: 1100
  - ```checkpoint_path```: path to save the model, i.e. ```../../checkpoint/geneExprModel_1.pt```
  - ```use_checkpoint```: ```True``` for loading the saved checkpoint and ```False``` for restart training

## Testing and Evaluation

### Test and evaluate the trained CPDM on the test set
```bash
python <VersioinName>.py <mode>
```
- Set ```<mode>``` to ```test```
- Refer to ```Training``` for parameter settings
- ```generateSamples```: number of images expected to generate
- ```gridW```: number of images on each row
- ```gridH```: number of images on each column
- ```figureSize```: size of generated images

### Generate images for some patients with unpaired data (patients only have genomic data)
```bash
python <VersioinName>.py <mode>
```
- Set ```<mode>``` to ```unpaired```
- Refer to ```Training``` and ```Test and evaluate the trained CPDM on the test set``` for parameter settings
- ```btfpath```: the file store all genomic data
- ```testIds```: a list of valid patient IDs

### Generate images for all patients with genomic data
```bash
python <VersioinName>.py <mode>
```
- Set ```<mode>``` to ```generate```
- Refer to ```Training``` and ```Test and evaluate the trained CPDM on the test set``` for parameter settings
- ```btfpath```: the file store all genomic data, i.e. ```../../allGeneExprs.csv```
- ```savePath```: the directory to save generated images, i.e. ```../../generatedImg_1```
- ```start```: the index of the first patient expected to generate images
- ```end```: the index of the last patient expected to generate images

# Applications

## Data collection and preprocessing

## Extract features
```bash
python extract.py path1 path2 path3
```
- ```path1```: path of file saving clinical data, i.e. mutations status, ER status, ER+/HER2+ subtypes. Eg. ```../../CliniUnOpHer2.csv```
- ```path2```: path of the folder saving generated MRI projections. Eg. ```../../innerResLoss2```
- ```path3```: path of the folder saving extracted features. Eg. ```../../0ri708er.pt```

## Predict TP53 mutation status

## Predict ER status

## Survival analysis based on the multi-omic profile-guided synthetic MRI projections

## Predict ER+/Her+2 subtype based on the multi-omic profile-guided synthetic MRI projections

## Predict ER+/Her+2 subtype based on the gene expression-guided synthetic MRI projections

## Survival analysis for patients with ER+/HER2+ subtype data (multi-omic version)

## Survival analysis for patients with ER+/HER2+ subtype data (gene expression version)


# Results






