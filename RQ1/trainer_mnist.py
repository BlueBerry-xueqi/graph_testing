import copy
import os
import os.path as osp
import pdb

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader

from models.geometric_models.mnist_nn_conv import Net as NN, train as MNIST_train, test as MNIST_test
from parser import Parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model: NN, dataset: MNIST

# construct the original model
def construct_model(args, dataset):
    if args.type == "NN":
        model = NN(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def load_data(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
    transform = T.Cartesian(cat=False)
    dataset = MNISTSuperpixels(path, True, transform=transform)
    dataset = dataset.shuffle()
    dataset = dataset[:5000]
    print("load data MNIST successfully!")

    n = len(dataset) // 10
    index = torch.randperm(len(dataset))
    train_index, val_index, test_index = index[:6 * n], index[6 * n:8 * n], index[8 * n:]
    # get dataset
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # save index path
    savedpath_index = f"pretrained_all/pretrained_model/{args.type}_{args.data}/index"
    if not os.path.isdir(savedpath_index):
        os.makedirs(savedpath_index, exist_ok=True)

    # save index
    torch.save(dataset, f"{savedpath_index}/dataset.pt")
    torch.save(train_index, f"{savedpath_index}/train_index.pt")
    torch.save(val_index, f"{savedpath_index}/val_index.pt")
    torch.save(test_index, f"{savedpath_index}/test_index.pt")

    return dataset, train_index, test_index, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def train_and_save(args):
    dataset, train_index, test_index, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data(
        args)

    # construct model
    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    best_acc = 0

    for epoch in range(args.epochs):
        MNIST_train(epoch, train_loader, model, optimizer)
        train_acc = MNIST_test(train_loader, train_dataset, model)
        val_acc = MNIST_test(val_loader, val_dataset, model)
        # select best test accuracy and save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}')
    print("Best val acuracy is: ", best_acc)

    # save best model
    savedpath_model = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_model):
        os.makedirs(savedpath_model, exist_ok=True)
    torch.save(best_model, os.path.join(savedpath_model, "model.pt"))

    # load best model, and get test accuracy
    model.load_state_dict(torch.load(os.path.join(savedpath_model, "model.pt"), map_location=device))
    mis_num = get_mis_num(test_loader, test_dataset, model)
    accuray = MNIST_test(test_loader, test_dataset, model)
    print("mis num: ", mis_num)
    print("accuray: ", accuray)
    print(len(test_index))

    # save best accuracy
    savedpath_num_misclassed = f"pretrained_all/misclassed_num/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_num_misclassed):
        os.makedirs(savedpath_num_misclassed, exist_ok=True)
    np.save(f"{savedpath_num_misclassed}/num.npy", mis_num)


def get_mis_num(test_loader, test_dataset, model):
    model.eval()
    correct = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    mis_num = len(test_dataset) - correct
    return mis_num


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
