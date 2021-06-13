from numpy import mod
from models.gin import train_and_save, construct_model
import torch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import os.path as osp
from test_metrics import sa

model_data_path="saved_model/gin_imdb_binary/"
#train_and_save(savedpath=model_data_path)

device = torch.device( "cuda:0" if torch.cuda.is_available()else "cpu" )
model = construct_model(device, savedpath=model_data_path)

path = osp.join('data', 'TU')
train_dataset, test_dataset = torch.load(f"{model_data_path}/train.pt"),torch.load(f"{model_data_path}/test.pt")
test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)

model.to(device)

#fetch_dsa(model, x_train_loader, x_target_loader, target_name, layer_names, **kwargs)
sa.fetch_dsa( model, train_loader, test_loader, "test", ["convs.3.nn.2", "convs.3.nn.3"], num_classes=2, is_classification=True,device=device,
        save_path="saved_model/gin_imdb_binary", d="IMDB", var_threshold=1e-5)


sa.fetch_lsa( model, train_loader, test_loader, "test", ["convs.3.nn.2", "convs.3.nn.3"], num_classes=2, is_classification=True,device=device,
        save_path="saved_model/gin_imdb_binary", d="IMDB", var_threshold=1e-5)

sa.fetch_sihoutete(model, train_loader, test_loader, "test", ["convs.3.nn.2", "convs.3.nn.3"], num_classes=2, is_classification=True,device=device,
        save_path="saved_model/gin_imdb_binary", d="IMDB", var_threshold=1e-5)

