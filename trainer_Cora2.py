import copy
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models.geometric_models.graph_unet import Net as UNET
from models.geometric_models.super_gat import Net as superGAT
from parser import Parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model
def construct_model(args, dataset):
    if args.type == "unet":
        model = UNET(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    if args.type == "superGAT":
        model = superGAT(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    return model, optimizer


# load dataset
def Cora_load_Data(args):
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset).shuffle()
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

    for epoch in range(args.epochs):
        Cora_train(model, optimizer, data, train_index)
        train_acc = Cora_test(model, data, train_index)
        val_acc = Cora_test(model, data, val_index)
        # select best test accuracy and save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}')
    print("Best val accuracy is: ", best_acc)

    # save best model
    savedpath_model = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_model):
        os.makedirs(savedpath_model, exist_ok=True)
    torch.save(best_model, os.path.join(savedpath_model, "model.pt"))

    # load best model, and get test accuracy
    model.load_state_dict(torch.load(os.path.join(savedpath_model, "model.pt"), map_location=device))
    test_acc = Cora_test(model, data, test_index)
    print("best test accuracy is: ", test_acc)

    # save best accuracy
    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy.npy", test_acc)


def Cora_train(model, optimizer, data, train_index):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_index], data.y[train_index]).backward()
    optimizer.step()


@torch.no_grad()
def Cora_test(model, data, index):
    model.eval()
    logits = model(data)
    pred = logits[index].max(1)[1]
    acc = pred.eq(data.y[index]).sum().item() / len(index)
    return acc


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
