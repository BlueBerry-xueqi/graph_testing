from scipy.stats.stats import mode
from test_metrics.ModelWrapper import NCModelWrapper
from numpy import mod
from models.gin import train_and_save, construct_model
import torch
from torch_geometric.transforms import OneHotDegree
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import os.path as osp
from test_metrics.nc import kmnc, nbc, tknc, nac

import scipy.stats as stats
model_data_path="saved_model/gin_imdb_binary/"
#train_and_save(savedpath=model_data_path)

device = torch.device( "cuda:0" if torch.cuda.is_available()else "cpu" )
model = construct_model(device, savedpath=model_data_path)

path = osp.join('data', 'TU')
train_dataset, test_dataset = torch.load(f"{model_data_path}/train.pt"),torch.load(f"{model_data_path}/test.pt")
test_loader = DataLoader(test_dataset, batch_size=1)
train_loader = DataLoader(train_dataset, batch_size=128)

model.to(device)
model.eval()
model_wrapper = NCModelWrapper(model, device)
layer_names = [ "convs.3.nn.2", "convs.3.nn.3" , "lin1"]
model_wrapper.register_layers(layer_names)
train,input,model, layers,k_bins=1000
kc = kmnc(train_loader, model_wrapper, layer_names)
h=kc.fit(test_loader)
print(h)
subset, scores= kc.rank_fast(test_loader)
print(scores.shape)
print(len(subset))

nb=nbc(train_loader, model_wrapper, layer_names, std=0.001)
h=nb.fit(test_loader)
print(h)
s,ss=nb.rank_fast(test_loader)
print(ss.shape)
print(len(s))
nb.rank_2(test_loader)


tk=tknc(test_loader, model_wrapper, layer_names)
s,ss=tk.rank(test_loader)
print(len(ss))
print(len(s))

nc=nac(test_loader, model_wrapper, layer_names, 0.01)
nc.fit()
nc.rank_2(test_loader)
nc.rank_fast(test_loader)