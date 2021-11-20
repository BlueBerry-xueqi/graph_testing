import math
import pdb

from transformers import get_cosine_schedule_with_warmup
from trainer_graph_classification import construct_model
import tqdm
from models.pytorchtools import EarlyStopping
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from parser import Parser
from models.data import get_dataset
from test_metrics.CES import CES_selection, selectsample
from test_metrics.DeepGini import deepgini_score
from test_metrics.random import random_select

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# get arguments
args = Parser().parse()

# get the ratio of the required selected data
select_ratio = int(args.select_ratio)


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


def loadData(args):
    model_name = args.type
    dataset_name = args.data
    savedpath = f"pretrained_all/pretrained_data/{model_name}_{dataset_name}"
    dataset = get_dataset(args.data,
                          normalize=args.normalize)  # TUDataset(path, name='COLLAB', transform=OneHotDegree(135)) #IMDB-BINARY binary classification

    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.8)
    testselection_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - testselection_size

    if not (os.path.isfile(f"{savedpath}/train_index.pt") and os.path.isfile(
            f"{savedpath}/test_selection_index.pt") and os.path.isfile(f"{savedpath}/test_index.pt")):
        print("ERROR===========The is no prepared dataset(train, test, test_selection)==========")
    else:
        train_dataset_index = torch.load(f"{savedpath}/train_index.pt")
        test_selection_index = torch.load(f"{savedpath}/test_selection_index.pt")
        test_dataset_index = torch.load(f"{savedpath}/test_index.pt")

        train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[
            test_selection_index], \
                                                              dataset[test_dataset_index]
        train_size = len(train_dataset_index)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
        test_selection_loader = DataLoader(test_selection_dataset, batch_size=100, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

        return dataset, train_loader, test_selection_loader, test_loader, test_selection_dataset, train_size, test_selection_index


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


def retrain(model, train_loader, metrics_select_data_loader, lr_schedule, train_size, select_num):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    if lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * train_size,
                                                    args.retrain_epochs * train_size)
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        if args.lr_schedule:
            scheduler.step()
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
    metrics = args.metrics

    dataset, train_loader, test_selection_loader, test_loader, test_selection_dataset, train_size, test_selection_index = loadData(
        args)

    for exp in range(0, args.exp):

            savedpath_retrain = f"retrained_all/retrained_model/{metrics}/{model_name}_{args.data}/{exp}_{epoch}/"
            if not os.path.isdir(savedpath_retrain):
                os.makedirs(savedpath_retrain, exist_ok=True)


            model = construct_model(args, dataset, model_name)
            model = model.to(device)
            model_path = f"{savedpath_retrain}retrained_model.pt"

            if os.path.isfile(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                savedpath_train = f"pretrained_all/pretrained_model/{model_name}_{args.data}_{epoch}_{exp}/"
                model.load_state_dict(torch.load(os.path.join(savedpath_train, "model.pt")))
                early_stopping = EarlyStopping(patience=args.patience, verbose=True, model_path=model_path)

                test_selection_size = len(test_selection_index)
                # get the selected number
                select_num = math.ceil(np.ceil(test_selection_size * select_ratio * 0.01))

                # dataset selected from test_selection_loader based on metrics
                metrics_select_data_loader = select_functions(model, test_selection_loader, test_selection_dataset,
                                                              select_num, metrics, ncl=None)
                for epoch in range(args.epochs):
                    retrain(model, train_loader, metrics_select_data_loader, args.lr_schedule, train_size, select_num)

                    torch.save(model.state_dict(), os.path.join(savedpath_retrain, "model.pt"))
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    savedpath_acc = f"retrained_all/retrain_accuracy/{model_name}_{args.data}/{epoch}_{exp}/"
                    if not os.path.isdir(savedpath_acc):
                        os.makedirs(savedpath_acc, exist_ok=True)
                    # get accuracy
                get_accuracy(model, test_loader, select_ratio, metrics, savedpath_acc)


# select retrain data based on metrics
def select_functions(model, target_loader, target_data, select_num, metric, ncl=None):
    if metric == "DeepGini":
        selected_index = deepgini_score(model, target_loader, select_num)
    elif metric == "random":
        selected_index = random_select(len(target_data), select_num)
    elif metric == "CES":
        selected_index = CES_selection(model, target_data, select_num)
    # if metric == "max_probability":
    #     scores, _, _ = max_score(model, target_loader)
    # if metric == "MCP":
    #     scores = MCP_score(model, target_loader, ncl)
    # # if metric == "ModelWrapper":
    # #     scores, _, _ = deepgini_score(model, target_loader)
    # if metric == "nc":
    #     scores, _, _ = nbc
    # if metric == "sa":
    #     scores, _, _ = deepgini_score(model, target_loader)
    # if metric == "sihoutete":
    #     scores, _, _ = deepgini_score(model, target_loader)

    # selected_index = np.argsort(scores)[:select_num]

    selected_loader = DataLoader(target_data[selected_index], batch_size=128)
    return selected_loader


def get_accuracy(model, test_loader, select_ratio, metrics, savedpath_acc):
    # create savedpath for retrained accuracy
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)

    # save file
    test_acc, test_loss = test(test_loader, model)
    np.save(f"{savedpath_acc}/{metrics}-{select_ratio}.npy", test_acc)


# accuracy when then dataset is selected randomly
if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
