import pdb

import tqdm

from models.pytorchtools import EarlyStopping
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
from parser import Parser
from models.data import get_dataset
from test_metrics.CES import selectsample
from test_metrics.DeepGini import deepgini_score
from test_metrics.MCP import MCP_score
from test_metrics.max_probability import max_score
from test_metrics.random import random_select
from trainer_graph_classification import construct_model
from test_metrics.nc import nbc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# get arguments
args = Parser().parse()

# get the ratio of the required selected data
select_ratio = int(args.select_ratio)

# get dataset
dataset = get_dataset(args.data,
                      normalize=args.normalize)
# get dataset size
dataset_size = len(dataset)

# get the selected number
select_num = int(dataset_size * select_ratio *0.01)


@torch.no_grad()
def test(loader, model):
    model.eval()
    total_correct = 0
    val_loss = 0

    for data in tqdm.tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        val_loss += loss.item() * data.num_graphs
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()

    return total_correct / len(loader.dataset), val_loss / len(loader.dataset)


# use metrics to select data based on metrics
def select_functions(model, target_loader, target_data, select_num, metric, savedpath, ncl=None):
    # if metric == "CES":
    #     scores, _, _ = selectsample(args, model, target_loader)

    if metric == "DeepGini":
        scores = deepgini_score(model, target_loader)
    if metric == "random":
        scores = random_select()
    # if metric == "max_probability":
    #     scores, _, _ = max_score(model, target_loader)
    # if metric == "MCP":
    #     scores, _, _ = MCP_score(self,test,use_lower=False)
    # if metric == "ModelWrapper":
    #     scores, _, _ = deepgini_score(model, target_loader)
    # if metric == "nc":
    #     scores, _, _ = nbc.rank_fast(model, target_loader)
    # if metric == "sa":
    #     scores, _, _ = deepgini_score(model, target_loader)
    # if metric == "sihoutete":
    #     scores, _, _ = deepgini_score(model, target_loader)

    # pdb.set_trace()
    # sort by score
    selected_index = np.argsort(scores)[:select_num]
    # save the index of selected dataset
    torch.save(selected_index, f"{savedpath}/selectedFunctionIndex.pt")
    selected_dataset = target_data[selected_index]
    selected_loader = DataLoader(selected_dataset, batch_size=128)

    return selected_loader


def loadData(args):
    model_name = args.type
    dataset_name = args.data
    savedpath = f"pretrained_all/pretrained_model/{model_name}_{dataset_name}"
    dataset = get_dataset(args.data,
                          normalize=args.normalize)  # TUDataset(path, name='COLLAB', transform=OneHotDegree(135)) #IMDB-BINARY binary classification

    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.8)
    # 选择的当作test data的数据的size是总数据集的0.1
    testselection_size = int(len(dataset) * 0.1)
    # 测试数据集的size是剩下的数据集
    test_size = len(dataset) - train_size - testselection_size

    # 假如说没有生成文件的前提下，生成文件
    if not (os.path.isfile(f"{savedpath}/train_index.pt") and os.path.isfile(
            f"{savedpath}/test_selection_index.pt") and os.path.isfile(f"{savedpath}/test_index.pt")):
        print("ERROR===========The is no prepared dataset(train, test, test_selection)==========")
    else:
        train_dataset_index = torch.load(f"{savedpath}/train_index.pt")
        test_selection_index = torch.load(f"{savedpath}/test_selection_index.pt")
        test_dataset_index = torch.load(f"{savedpath}/test_index.pt")

    train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[test_selection_index], \
                                                          dataset[test_dataset_index]

    train_size = len(train_dataset_index)
    # test loader是测试数据集
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    test_selection_loader = DataLoader(test_selection_dataset, batch_size=100, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    return train_loader, test_selection_loader, test_loader, test_selection_dataset, train_size


@torch.no_grad()
def test(loader, model):
    model.eval()
    total_correct = 0
    val_loss = 0

    for data in tqdm.tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        val_loss += loss.item() * data.num_graphs
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()
    return total_correct / len(loader.dataset), val_loss / len(loader.dataset)


def retrain(model, test_selection_loader, test_selection_dataset, lr_schedule, train_size, select_num):
    model_name = args.type
    dataset_name = args.data
    metrics = args.metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * train_size,
                                                    args.num_epochs * train_size)
    model.train()
    total_loss = 0.0

    savedpath_metrics = f"retrained_all/metrics_select_loader/{model_name}_{dataset_name}_{metrics}/selected_data.pt"
    if not os.path.isdir(savedpath_metrics):
        os.makedirs(savedpath_metrics, exist_ok=True)

    # dataset selected from test_selection_loader based on metrics
    metrics_select_data_loader = select_functions(model, test_selection_loader, test_selection_dataset,
                                                  select_num, metrics, savedpath_metrics, ncl=None)
    # pdb.set_trace()
    # retrain model using metrics to select
    for data in metrics_select_data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        if args.lr_schedule:
            scheduler.step()

    return total_loss / (train_size + select_num)


def retrain_and_save(args):
    # get model name and type
    model_name = args.type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = args.data
    metrics = args.metrics

    # get dataset
    savedpath_train = f"pretrained_all/pretrained_model/{model_name}_{args.data}"
    if not os.path.isdir(savedpath_train):
        print("============== savedpath not exits==============")

    dataset = get_dataset(args.data,
                          normalize=args.normalize)
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.8)

    train_dataset_index = torch.load(f"{savedpath_train}/train_index.pt")
    test_selection_index = torch.load(f"{savedpath_train}/test_selection_index.pt")
    test_dataset_index = torch.load(f"{savedpath_train}/test_index.pt")

    train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[
        test_selection_index], dataset[test_dataset_index]

    # rank and select data for retraining
    test_selection_loader = DataLoader(test_selection_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # savedpath retrain
    savedpath_retrain = f"retrained_all/retrained_model/{metrics}/{model_name}_{args.data}/"
    if not os.path.isdir(savedpath_retrain):
        os.makedirs(savedpath_retrain, exist_ok=True)

    # get basedline model
    model = construct_model(args, dataset, device, savedpath_train, model_name)
    model.load_state_dict(torch.load(f"pretrained_all/pretrained_model/{model_name}_{data}/pretrained_model.pt"))
    model = model.to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=savedpath_retrain)

    for epoch in range(args.retrain_epochs):
        # get select loader by metrics
        loss = retrain(model, test_selection_loader, test_selection_dataset, args.lr_schedule, train_size, select_num)
        test_acc, test_loss = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ')
        early_stopping(test_loss, model, performance={"test_acc": test_acc, "test_loss": test_loss})
        if early_stopping.early_stop:
            print("******** Early stopping")
            break
    model_name = args.type
    dataset_name = args.data

    # get accuracy
    get_accuracy(model, test_loader, model_name, dataset_name, select_num, metrics)


# select retrain data based on metrics
def select_functions(model, target_loader, target_data, select_num, metric, savedpath, ncl=None):
    if metric == "DeepGini":
        scores = deepgini_score(model, target_loader)
    if metric == "random":
        scores = random_select()
    if metric == "CES":
        scores = selectsample(args, model, target_loader)
    if metric == "max_probability":
        scores, _, _ = max_score(model, target_loader)
    if metric == "MCP":
        scores = MCP_score(model, target_loader, ncl)
    # if metric == "ModelWrapper":
    #     scores, _, _ = deepgini_score(model, target_loader)

    if metric == "nc":
        scores, _, _ = nbc
    if metric == "sa":
        scores, _, _ = deepgini_score(model, target_loader)
    if metric == "sihoutete":
        scores, _, _ = deepgini_score(model, target_loader)

    selected_index = np.argsort(scores)[:select_num]
    torch.save(selected_index, f"{savedpath}/selectedFunctionIndex_{select_num}.pt")
    selected_loader = DataLoader(target_data[selected_index], batch_size=128)
    return selected_loader


def get_accuracy(model, test_loader, model_name, dataname, select_num, metrics):
    # create savedpath for retrained accuracy
    savedpath_retrain_accuracy = f"retrained_all/retrain_accuracy/{metrics}/{model_name}_{dataname}_{select_ratio}"
    if not os.path.isdir(savedpath_retrain_accuracy):
        os.makedirs(savedpath_retrain_accuracy, exist_ok=True)

    # save file
    test_acc, test_loss = test(test_loader, model)
    with open(f"{savedpath_retrain_accuracy}/retrain_accuracy_3.txt", 'a') as f:
        f.write(str(test_acc) + "\n")

# accuracy when then dataset is selected randomly

if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
