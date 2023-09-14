
python retraining_main.py --cuda 'cuda:1' --model_name 'gcn' --target_model_path './target_models/cora_gcn.pt' --path_x '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --path_save './result/cora_gcn.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'gat' --target_model_path './target_models/cora_gat.pt' --path_x '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --path_save './result/cora_gat.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'graphsage' --target_model_path './target_models/cora_graphsage.pt' --path_x '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --path_save './result/cora_graphsage.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'tagcn' --target_model_path './target_models/cora_tagcn.pt' --path_x '../data/cora/x_np.pkl' --path_edge_index '../data/cora/edge_index_np.pkl' --path_y '../data/cora/y_np.pkl' --path_save './result/cora_tagcn.json'

python retraining_main.py --cuda 'cuda:1' --model_name 'gcn' --target_model_path './target_models/citeseer_gcn.pt' --path_x '../data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --path_save './result/citeseer_gcn.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'gat' --target_model_path './target_models/citeseer_gat.pt' --path_x '../data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --path_save './result/citeseer_gat.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'graphsage' --target_model_path './target_models/citeseer_graphsage.pt' --path_x '../data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --path_save './result/citeseer_graphsage.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'tagcn' --target_model_path './target_models/citeseer_tagcn.pt' --path_x '../data/citeseer/x_np.pkl' --path_edge_index '../data/citeseer/edge_index_np.pkl' --path_y '../data/citeseer/y_np.pkl' --path_save './result/citeseer_tagcn.json'

python retraining_main.py --cuda 'cuda:1' --model_name 'gcn' --target_model_path './target_models/pubmed_gcn.pt' --path_x '../data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --path_save './result/pubmed_gcn.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'gat' --target_model_path './target_models/pubmed_gat.pt' --path_x '../data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --path_save './result/pubmed_gat.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'graphsage' --target_model_path './target_models/pubmed_graphsage.pt' --path_x '../data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --path_save './result/pubmed_graphsage.json'
python retraining_main.py --cuda 'cuda:1' --model_name 'tagcn' --target_model_path './target_models/pubmed_tagcn.pt' --path_x '../data/pubmed/x_np.pkl' --path_edge_index '../data/pubmed/edge_index_np.pkl' --path_y '../data/pubmed/y_np.pkl' --path_save './result/pubmed_tagcn.json'
