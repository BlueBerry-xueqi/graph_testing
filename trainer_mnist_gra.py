import copy
import os
import os.path as osp

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader

from models.geometric_models.mnist_graclus import Net as graclus, train as graclus_train, test as graclus_test
from parser import Parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model: NN, dataset: MNIST

# construct the original model
def construct_model(args, dataset):
    if args.type == "gra":
        model = graclus(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def load_data(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
    transform = T.Cartesian(cat=False)
    dataset = MNISTSuperpixels(path, True, transform=transform)
    dataset = dataset.shuffle()
    dataset = dataset[:10000]
    print("load data MNIST successfully!")

    n = len(dataset) // 10
    index = torch.randperm(len(dataset))
    train_index, val_index, test_index, retrain_index = index[:4 * n], index[4 * n:5 * n], index[5 * n: 6 * n], index[
                                                                                                                6 * n:]
    # get dataset
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    retrain_dataset = dataset[retrain_index]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    retrain_loader = DataLoader(retrain_dataset, batch_size=64, shuffle=False)

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

    return dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


def train_and_save(args):
    dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = load_data(
        args)

    # construct model
    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    best_acc = 0
    best_val_acc = test_acc = 0

    for epoch in range(args.epochs):
        graclus_train(epoch, model, optimizer, train_loader)
        train_acc = graclus_test(model, train_loader, train_dataset)
        val_acc = graclus_test(model, val_loader, val_dataset)
        tmp_test_acc = graclus_test(model, test_loader, test_dataset)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}')
    print(best_acc)

    savedpath_model = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_model):
        os.makedirs(savedpath_model, exist_ok=True)
    torch.save(best_model, os.path.join(savedpath_model, "model.pt"))

    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy.npy", best_acc)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
