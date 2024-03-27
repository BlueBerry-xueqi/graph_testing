# Source Code of Graph Testing
## Main Requirements
- PyTorch 
- PyTorch Geometric
- scikit-learn
- networkx

## Dataset
- Cora  https://graphsandnetworks.com/the-cora-dataset
- CiteSeer  https://paperswithcode.com/dataset/citeseer
- PubMed    https://paperswithcode.com/dataset/pubmed
- Mutagenicity  https://paperswithcode.com/dataset/mutagenicity
- NCI1  https://paperswithcode.com/dataset/nci1
- GraphMNIST    https://huggingface.co/datasets/graphs-datasets/MNIST
- MSRC21    https://chrsmrrs.github.io/datasets/docs/datasets/
- DrugBank  https://tdcommons.ai/multi_pred_tasks/ddi/
- BindingDB https://tdcommons.ai/multi_pred_tasks/dti/
## cluster_metrics

## cluster_uncertaity_retraining

## edge 
The file 'edge' contains code for the edge task, including misclassification detection and accuracy estimation.
#### How to run misclassification detection
    python misclassification.py
#### How to run accuracy estimation
    python estimation.py --path_model './edge_model/BindingDB_graphsageEdge.pt' --path_x './data/BindingDB/x_np.pkl' --path_edge_index './data/BindingDB/edge_index_np.pkl' --path_y './data/BindingDB/y_np.pkl' --model_name 'graphsageEdge' --save_path './result/estimation_2_BindingDB_graphsageEdge.json' --epochs 10

## estimation

## misclassification

## models/geometric_models

## nodelmportance_retraining
## results


