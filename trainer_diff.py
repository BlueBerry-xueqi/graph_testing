import copy
import os
import os.path as osp
import pdb

import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T

from models.geometric_models.proteins_diff_pool import Net as diff, train as diff_train, test as diff_test
from parser import Parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model
def construct_model(args, dataset):
    if args.type == "diff":
        model = diff(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def load_data(args):
    max_nodes = 150

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                    'PROTEINS_dense')
    dataset = TUDataset(path, name=args.data, transform=T.ToDense(max_nodes),
                        )
    dataset = dataset.shuffle()

    n = len(dataset) // 10
    index = torch.randperm(len(dataset))
    train_index, val_index, test_index, retrain_index = index[:4 * n], index[4 * n:5 * n], index[5 * n: 6 * n], index[
                                                                                                                6 * n:]
    # get dataset
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    retrain_dataset = dataset[retrain_index]

    train_loader = DenseDataLoader(train_dataset, batch_size=60, shuffle=True)
    val_loader = DenseDataLoader(val_dataset, batch_size=60, shuffle=False)
    test_loader = DenseDataLoader(test_dataset, batch_size=60, shuffle=False)
    retrain_loader = DenseDataLoader(retrain_dataset, batch_size=60, shuffle=False)

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

    return dataset, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


def train_and_save(args):
    # load data
    dataset, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = load_data(
        args)
    # construct model
    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    # train model in many epochs
    best_acc = 0
    for epoch in range(args.epochs):
        # train model
        diff_train(model, optimizer, train_loader, train_dataset)
        # get test accuracy of model
        train_acc = diff_test(model, train_loader)
        val_acc = diff_test(model, val_loader)

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
    test_acc = diff_test(model, test_loader)
    print("best test accuracy is: ", test_acc)

    # save best accuracy
    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy.npy", test_acc)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
