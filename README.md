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
```python
python <VersioinName>.py <mode>
```
- ```<VersioinName>``` represents the CPDM trained on the different dataset, which can be ```multiOmicVersion``` and ```geneExprVersion```
- if user only want to train a CPDM and save the model, please set ```<mode>``` to ```train```
- Please set correct parameters in the , including:
  - SideImgPath: the directory store real MRI projections, i.e. ```../../SidePNG128```
  - SideBTFpath = '/content/drive/MyDrive/End2End/SideViewBTF.csv'
  - allBTFpath = '/content/drive/MyDrive/End2End/BTF_features.csv'
  - geneExprPath = '/content/drive/MyDrive/geneExpr/kyle/TCGABRCA_15gxp.csv'
  - mutationStatusPath = '/content/drive/MyDrive/End2End/all_genes.csv'
  - clinicalDataPath = '/content/drive/MyDrive/diffED/CliniUnOpHer2.csv'
  - image_size = 128
  - channels = 1
  - timesteps = 1500
  - loss_type = 'l2' # l1
  - lr = 1e-4
  - batch_size = 6  # 6
  - epochs = 1100  # 1100
  - gene = 'TP53'
  - device = 'cuda' if torch.cuda.is_available() else 'cpu'
  - checkpoint_path = '/content/drive/MyDrive/geneExpr/checkpoint/geneExprModel_1.pt' 
  - use_checkpoint = False  # True for load
## Testing

To test a trained






```python
python matchimage.py --csvfile <file> --imagedir <image_directory> --outputdir <output_directory>
```
```bash
sssss
```

