# MMT-fgMDI_LWZ
MMT-fgMDI: Metapath-driven multimodal transformer framework for predicting fine-grained metabolite-drug interactions

## 🏠 Overview
![flow_chart](https://github.com/user-attachments/assets/4afa488c-047b-495a-a41e-9d4ecd7b9f20)

## 🛠️ Dependecies
```
- conda=24.4.0
- Python == 3.12
- pytorch == 2.3.0+cu121
- torch_geometric == 2.5.3
- numpy == 1.26.4
- pandas == 2.2.2
- scikit-learn == 1.5.0
- matplotlib == 3.9.0
- GPU == NVIDIA RTX 4090D (24G) GPUs * 1
- CPU == 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
```

## 🗓️ Dataset
###  
```
- drug-metabolite upregulation network: Dr-Up-M.csv
- drug-metabolite downregulation network: Dr-Down-M.csv
- drug-metabolite assciations: Dr-M.csv
- drug-disease assciations: Dr-Di.csv
- drug-gene assciations: Dr-G.csv
- metabolite-disease assciations: M-Di.csv
- metabolite-gene assciations: M-G.csv
- drug initial feature: drug_mol2vec.csv
- metabolite initial feature: metabolite_mol2vec.csv
- disease initial feature: Mesh2vec.csv
- gene initial feature: DNA2vec.csv
```
###   multimodal networks 
![multimodal networks](https://github.com/user-attachments/assets/dce6e7f4-2a73-4e7f-bd90-da7614eba192)



## 🛠️ Model options
###  training parameters
```
--seed             int     Random seed                                Default is 0.
--epochs           int     Number of training epochs.                 Default is 1000.
--weight_decay     float   Weight decay                               Default is 5e-4.
--dropout          float   Dropout rate                               Default is 0.1.
--lr               float   Learning rate                              Default is 0.01.
```

###  model parameters

![model parameters](https://github.com/user-attachments/assets/52188fe0-3940-42fa-aa63-884b47e1c489)



## 🎯 How to run?
```
1. The data1 and data2 folders store the association networks, disease and metabolite networks, and initial characterization data for datasets 1 and 2, respectively.
2. The code folder for implementing the RGCGT model, which specifically includes:
  (1) train.py is used to start the RGCGT model and set up parameters, implement training and validation, loss function definition, optimizer selection and parameter update.
  (2) model.py is used to build the overall structure of the RGCGT model, including residual graph convolution (RGC), graph transformer (GT) with multi-hop neighbor aggregation and decoder.
  (3) layers.py mainly stores some customized network layers, including multi-head self-attention layer and feed-forward network, etc.
  (4) utils.py mainly realizes data loading, evaluation index calculation and plot, etc.
3. The results folder stores detailed experimental results on datasets 1 and 2 using jupyter notebook.
```
