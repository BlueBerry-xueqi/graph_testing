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

## File 

### cluster_metrics
The file cluster_metrics contains all the code for clustering-based test selection metrics.

### cluster_uncertaity_retraining
The file cluster_uncertainty_retraining contains all the code for RQ3 (GNN model retraining), including using the misclassification detection methods and accuracy estimation method for retraining. 
#### How to run 
    python selection_[#datasetName].py --type [model name] --metrics [selection method name] --select_ratio [#selection ratio] --exp [#number] --retrain_epochs [#number]
    We provide an example below. 
    python selection_PubMed.py --type ARMA --metrics deepgini --select_ratio 5 --exp 3 --retrain_epochs 20

### edge 
The file 'edge' contains code for the edge task, including misclassification detection and accuracy estimation.
#### How to run misclassification detection
    python misclassification.py
#### How to run accuracy estimation
    python estimation.py --path_model './edge_model/BindingDB_graphsageEdge.pt' --path_x './data/BindingDB/x_np.pkl' --path_edge_index './data/BindingDB/edge_index_np.pkl' --path_y './data/BindingDB/y_np.pkl' --model_name 'graphsageEdge' --save_path './result/estimation_2_BindingDB_graphsageEdge.json' --epochs 10

### estimation
The file cluster_uncertainty_retraining contains all the code for RQ2 (GNN accuracy estimation). 
#### How to run 
    We provide an example below. 
    python selection_PubMed.py --type AGNN --metrics kmeans --select_ratio 5 --exp 3 --retrain_epochs 20



### misclassification
The file cluster_uncertainty_retraining contains all the code for RQ1 (GNN misclassification detection). 
#### How to run 
     We provide an example below. 
     python selection_Cora.py --type GCN --metrics kmeans --select_ratio 5 --exp 3 --retrain_epochs 20



### models/geometric_models
The file includes the GNN model used to evaluate the test selection methods.

### nodelmportance_retraining

### results
The file contains the experimental results for the empirical study.



