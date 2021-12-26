import copy
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from models.geometric_models.AGNN import Net as AGNN
from models.geometric_models.ARMA import Net as ARMA
from models.geometric_models.GAT import Net as GAT
from models.geometric_models.GCN_model import Net as GCN
from parser import Parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model
def construct_model(args, dataset):
    if args.type == "GCN":
        model = GCN(dataset.num_features, dataset.num_classes)

        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)

    elif args.type == "GAT":
        model = GAT(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    elif args.type == "AGNN":
        model = AGNN(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    elif args.type == "ARMA":
        model = ARMA(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    return model, optimizer


# load dataset
def Cora_load_Data(args):
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)

    # split data : train:val:test;retrain = 4:1:1:4
    n = int(len(data.y) / 10)
    index = torch.randperm(len(data.y))
    train_index, val_index, test_index, retrain_index = index[:4 * n], index[4 * n:5 * n], index[5 * n: 6 * n], index[
                                                                                                                6 * n:]

    # save index path
    savedpath_index = f"pretrained_all/pretrained_model/{args.type}_{args.data}/index"
    if not os.path.isdir(savedpath_index):
        os.makedirs(savedpath_index, exist_ok=True)

    # save index
    torch.save(dataset, f"{savedpath_index}/dataset.pt")
    torch.save(train_index, f"{savedpath_index}/train_index.pt")
    torch.save(val_index, f"{savedpath_index}/val_index.pt")
    torch.save(test_index, f"{savedpath_index}/test_index.pt")
    torch.save(retrain_index, f"{savedpath_index}/retrain_index.pt")
    return data, dataset, train_index, val_index, test_index, retrain_index


def train_and_save(args):
    # load dataset
    data, dataset, train_index, val_index, test_index, retrain_index = Cora_load_Data(args)
    # construct model
    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    best_acc = 0
    best_val_acc = test_acc = 0

    for epoch in range(args.epochs):
        Cora_train(model, optimizer, data, train_index)
        train_acc, val_acc, tmp_test_acc = Cora_test(model, data, train_index, val_index, test_index)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}')
    print(best_acc)

    # save best model
    savedpath_pretrain = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_pretrain):
        os.makedirs(savedpath_pretrain, exist_ok=True)
    torch.save(best_model, os.path.join(savedpath_pretrain, "model.pt"))

    # save best accuracy
    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy.npy", best_acc)


def Cora_train(model, optimizer, data, train_index):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_index], data.y[train_index]).backward()
    optimizer.step()


@torch.no_grad()
def Cora_test(model, data, train_index, val_index, test_index):
    list_index = [train_index, val_index, test_index]
    model.eval()
    logits, accs = model(data), []
    for index in list_index:
        pred = logits[index].max(1)[1]
        acc = pred.eq(data.y[index]).sum().item() / len(index)
        accs.append(acc)
    return accs


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
