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

#

device = torch.device( "cuda:0" if torch.cuda.is_available()else "cpu" )
kl=[]
nl=[]
tl=[]
al=[]

def run():
    model_data_path="saved_model/gin_imdb_binary/"
    train_and_save(savedpath=model_data_path)
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

    kc = kmnc(train_loader, model_wrapper, layer_names)
    h=kc.fit(test_loader)
    # print(h)
    subset, scores,pre, ground= kc.rank_fast(test_loader)
    tau, p_value = stats.kendalltau(scores, pre != ground)
    res = []
    print(f"KMNC Kendall Tau {tau}, Pvalue {p_value}")
    res.append(f"KMNC Kendall Tau {tau}, Pvalue {p_value}")
    kl.append(tau)
    nb=nbc(train_loader, model_wrapper, layer_names, std=0.001)
    s,ss,pre, ground=nb.rank_fast(test_loader)
    tau, p_value = stats.kendalltau(ss, pre != ground)
    print(f"NBC Kendall Tau {tau}, Pvalue {p_value}")
    nl.append(tau)
    res.append(f"NBC Kendall Tau {tau}, Pvalue {p_value}")

    tk=tknc(test_loader, model_wrapper, layer_names)
    s,ss,pre, ground=tk.rank(test_loader)
    tau, p_value = stats.kendalltau(ss, pre != ground)
    print(f"TKNC Kendall Tau {tau}, Pvalue {p_value}")
    tl.append(tau)
    res.append(f"TKNC Kendall Tau {tau}, Pvalue {p_value}")
    nc=nac(test_loader, model_wrapper, layer_names, 0.01)
    nc.fit()
    s,ss, pre, ground = nc.rank_2(test_loader)
    tau, p_value = stats.kendalltau(ss, pre != ground)
    print(f"NAC Kendall Tau {tau}, Pvalue {p_value}")
    al.append(tau)
    res.append(f"NAC Kendall Tau {tau}, Pvalue {p_value}")
    s,ss, pre, ground= nc.rank_fast(test_loader)
    tau, p_value = stats.kendalltau(ss, pre != ground)
    print(f"NAC Kendall Tau {tau}, Pvalue {p_value}")
    res.append(f"NAC Kendall Tau {tau}, Pvalue {p_value}")
    print(res)

import numpy as np
for i in range(10):
    run()
    
print("KMNC {}, {}".format(np.mean(kl), np.var(kl)))
print("NBC {}, {}".format(np.mean(nl), np.var(nl)))
print("TKNC {}, {}".format(np.mean(tl), np.var(tl)))
print("NAC {}, {}".format(np.mean(al), np.var(al)))