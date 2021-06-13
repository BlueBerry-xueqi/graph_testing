from numpy import mod
from models.gin import Net, construct_model
import torch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import os.path as osp
from test_metrics import sa

device = torch.device( "cuda:0" if torch.cuda.is_available()else "cpu" )
model = construct_model(device)

path = osp.join('data', 'TU')
dataset = TUDataset(path, name='IMDB-BINARY', transform=OneHotDegree(135)) #IMDB-BINARY binary classification
dataset = dataset.shuffle()
train_size = int(len(dataset)*0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split( dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
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

