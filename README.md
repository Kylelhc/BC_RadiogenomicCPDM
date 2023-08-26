# Conditional Probabilistic Diffusion Model Driven Synthetic Radiogenomic Applications in Breast Cancer

# Introduction
This repository provides the source codes and raw datasets associated with the paper Conditional Probabilistic Diffusion Model (CPDM) Driven Synthetic Radiogenomic Applications in Breast Cancer (BC).

BC exhibits significant heterogeneity, making it crucial to analyze its phenotypic diversity at a multi-omics level for early detection and personalized treatment. The integration of medical images and genomics offers a novel perspective on the study of BC heterogeneity. However, the absence of paired medical images and genomics data could pose a significant challenge.

We proposed the utilization of a well-trained CPDM to address the unpaired data issue in radiogenomic study of BC. The generated images will then be used to predict clinical attributes including gene mutations, estrogen receptor (ER) status, human epidermal growth factor receptor 2-positive (ER+/HER2+) status and have survival significance. The overall project workflow is depicted below.

![workflow](https://github.com/Kylelhc/BC_RadiogenomicCPDM/assets/143105097/39ce3ab5-733e-42bd-be3e-efeb22ce97a6)

# CPDM
## Architecture
![Fig  1](https://github.com/Kylelhc/BC_RadiogenomicCPDM/assets/143105097/922c1fb2-32fb-4f89-91bd-b8e0d75356dc)

```python
python matchimage.py --csvfile <file> --imagedir <image_directory> --outputdir <output_directory>
```
```bash
sssss
```

