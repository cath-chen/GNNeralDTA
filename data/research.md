# [Davis Dataset](https://github.com/hkmztrk/DeepDTA/tree/master/data)

## Introduction slide
- Type(s) of data:\
    Compounds(drugs), proteins, and their binding affinities
  
- Source of data:\
    Experiment results published by Davis et al. (2011)

- Sample size:\
    \# of proteins: 442\
    \# of drugs: 68\
    Binding entities: 30,056

- Features:
    - Drug-Target affinity (DTA) information
    - Suitable for graph model design

## Papers

- [DeepDTA: deep drug–target binding affinity prediction](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245)
    - 2018, MSE 0.261, CI 0.878
    - reference paper
    - label encoding with dictionary
    - Keras embedding layer (TODO: what's Keras)
    - 1D-CNN with 3 layers and MaxPool
    - Concatenate SMILES and Sequence
    - FC layers for output
- [Data for real-valued drug-target binding affinity prediction experiments](https://staff.cs.utu.fi/~aatapa/data/DrugTarget/)
    - original Davis data
- [SimBoost: a read-across approach for predicting drug–target binding affinities using gradient boosting machines](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z)
    - original KIBA data
- [AttentionDTA: prediction of drug–target binding affinity using attention model](https://ieeexplore.ieee.org/abstract/document/8983125)
    - 2019, MSE 0.215, CI 0.893
    - like DeepDTA, but attention mechanism to combine sequences
- [GEFA: Early Fusion Approach in Drug-Target Affinity Prediction](https://ieeexplore.ieee.org/abstract/document/9470924)
    - 2022, MSE 0.228, CI 0.893
    - uses graph model and 2D structure of protein (protein contact map TODO: what's that)
    - refine drug graph with GCN, fuse graphs, refine
- [WideDTA: prediction of drug-target binding affinity](https://arxiv.org/abs/1902.04166)
    - 2019, MSE 0.262, CI 0.886 
    - extract words of length 8 from the SMILES data, instead of character based encoding
- [SSM-DTA: Breaking the Barriers of Data Scarcity in Drug-Target Affinity Prediction](https://arxiv.org/abs/2206.09818)
    - 2022, MSE 0.219, 0.890
    - encode drugs and proteins with transformers
    - perform cross-attention and final MLP for prediction
    - to improve encoding considering limited data use multi-task learning: mask part of input sequence and try to reconstruct sequence from encoding in parallel to affinity prediction
    - also employ unlabeled data to improve reconstruction
- [Affinity2Vec: drug-target binding affinity prediction through representation learning, graph mining, and machine learning](https://www.nature.com/articles/s41598-022-08787-9)
    - 2022, MSE 0.24, CI 0.887
    - generate graph containing drugs and targets with edges between all combinations as similarity/affinity scores, filtered by threshold values
    - seq2seq for drug encoding, ProtVec for protein encoding
    - use CosSim to embed new data into graph
    - perform graph mining to interpolate new scores, combine with embedding model
- [Deep drug-target binding affinity prediction with multiple attention blocks](https://academic.oup.com/bib/article/22/5/bbab117/6231754) (MATT_DTI)
    - 2021, MSE 0.229, CI 0.890 
    - token and position embedding of SMILES are fed into relative self-attention before CNN like in DeepDTA
    - multi-head attention to compare drug/protein representation, rest like DeepDTA
- [FusionDTA: attention-based feature polymerizer and knowledge distillation for drug-target binding affinity prediction](https://academic.oup.com/bib/article/23/1/bbab506/6470967)
    - 2021, MSE 0.208, CI 0.913
    - transformer for protein embedding, pre-train by predicting masked parts
    - embedding matrix for drugs
    - both passed through feedforward and then two two-layer bidirectional LSTM
    - Fusion layer performs multi-head attention, once only drug, once only target, once both
    - knowledge distillation to reduce parameters
- [GraphDTA: predicting drug–target binding affinity with graph neural networks](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970)
    - 2020, MSE 0.229, CI 0.893
    - SMILES to molecular graph with RDKit, each atom has information stored
        - atom symbol, number of adjacent atoms, number of adjacent hydrogens, implicit value of atom, whether aromatic structure
    - compares GCN, GAT, GIN, GAT + GCN
        - applies max pooling for output
        - GIN performs best, GAT close second
- [MGraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d1sc05180f)
    - 2022, MSE 0.204, CI 0.900
    - multiscale GNN
        - multiscale blocks with dense connections between convolution layers
        - transition layers connect adjacent multiscale blocks
        - average pooling for final output
    - multiscale CNN for protein
    - concat and MLP for DTA prediction
  
## Data

- affinity measured in _kinase dissociation constant_ $K_d$ range (5.0, 10.8)
    $$pK_d = - \log_{10}\left(\frac{K_{d}}{1e^9}\right)$$
- drugs as SMILES sequence, maximum length 103, average 64 (TODO what's SMILES)
    - sequence follows a main branch
    - side branches in braces
    - loops are cut and indexed, indices inserted at each end of cut
- protein sequence length max. 2,549, avg. 788
- provided with 5-fold, distinct test set
- concordance index (CI) measures how many of all possible pairs are values are ordered correctly