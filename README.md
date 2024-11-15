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
- centrality feature of drug-disease association network: Dr-Di_centrality.csv
- centrality feature of drug-gene association network: Dr-G_centrality.csv
- centrality feature of metabolite-disease association network: M-Di_centrality.csv
- centrality feature of metabolite-gene association network: M-G_centrality.csv

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
1. The folder of the dataset stores various associations and initial feature information..
2. The graph_transformer folder for implementing the MMT-fgMDI model, which specifically includes:
  (1) train.py is used to start the MMT-fgMDI model and set up parameters, implement training and validation, loss function definition, optimizer selection and parameter update.
  (2) model.py is used to build the overall structure of the MMT-fgMDI model, including metapath-driven multimodal transformer encoder and decoder.
  (3) layers.py mainly stores some customized network layers, including multi-head self-attention layer and feed-forward network, etc.
  (4) utils.py mainly realizes data loading, confusion matrix and t-sne visualization, etc.
3. The results folder stores detailed experimental results.
```
