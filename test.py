from scipy.stats.stats import mode
from test_metrics.ModelWrapper import SA_Model
from numpy import mod
from models.gin import train_and_save, construct_model
import torch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import os.path as osp
from test_metrics import sa 
from test_metrics import sihoutete
from test_metrics import DeepGini , MCP
import scipy.stats as stats
model_data_path="saved_model/gin_imdb_binary/"
#train_and_save(savedpath=model_data_path)

device = torch.device( "cuda:0" if torch.cuda.is_available()else "cpu" )
model = construct_model(device, savedpath=model_data_path)

path = osp.join('data', 'TU')
train_dataset, test_dataset = torch.load(f"{model_data_path}/train.pt"),torch.load(f"{model_data_path}/test.pt")
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)

model.to(device)
model.eval()
model_wrapper = SA_Model(model, device)
layer_names = [ "convs.3.nn.2", "convs.3.nn.3" ]
model_wrapper.register_layers(layer_names)

#fetch_dsa(model, x_train_loader, x_target_loader, target_name, layer_names, **kwargs)
dsa, target_pred_dsa, ground_truth_tagrget = sa.fetch_dsa( model_wrapper, train_loader, test_loader, "test",layer_names, num_classes=2, is_classification=True,
        save_path="saved_model/gin_imdb_binary", d="IMDB", var_threshold=1e-5)
tau, p_value = stats.kendalltau(dsa, ground_truth_tagrget != target_pred_dsa)

print(f"DSA Kendall Tau {tau}, Pvalue {p_value}")


lsa, target_pred_lsa, ground_truth_tagrget = sa.fetch_lsa( model_wrapper, train_loader, test_loader, "test",layer_names, num_classes=2, is_classification=True,
        save_path="saved_model/gin_imdb_binary", d="IMDB", var_threshold=1e-5)
tau, p_value = stats.kendalltau(lsa, ground_truth_tagrget!= target_pred_lsa)
print(f"LSA Kendall Tau {tau}, Pvalue {p_value}")

si, target_pred,ground_truth_tagrget = sihoutete.fetch_sihoutete(model_wrapper, train_loader, test_loader, "test", ["convs.3.nn.2", "convs.4.nn.3", ], num_classes=2, is_classification=True,device=device,
        save_path="saved_model/gin_imdb_binary", d="IMDB", var_threshold=1e-5)
tau, p_value = stats.kendalltau(si, ground_truth_tagrget!=target_pred)
print(f"Si Kendall Tau {tau}, Pvalue {p_value}")


gigniscore, pre_labels, ground_truth  = DeepGini.deepgini_score(model_wrapper, test_loader)
tau, p_value = stats.kendalltau(gigniscore, ground_truth!=pre_labels)
print(f"DeepGiNi Kendall Tau {tau}, Pvalue {p_value}")


mcpscore, pre_labels, ground_truth  = MCP.margin_score(model_wrapper, test_loader)
tau, p_value = stats.kendalltau(mcpscore, ground_truth!=pre_labels)
print(f"MCP Kendall Tau {tau}, Pvalue {p_value}")



maxscore, pre_labels, ground_truth  = MCP.margin_score(model_wrapper, test_loader)
tau, p_value = stats.kendalltau(maxscore, ground_truth!=pre_labels)
print(f"MaxP Kendall Tau {tau}, Pvalue {p_value}")

