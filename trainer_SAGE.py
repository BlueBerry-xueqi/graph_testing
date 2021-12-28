import copy
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader

from models.geometric_models.mem_pool import train as mem_train, test as mem_test
from models.geometric_models.mnist_nn_conv import train as MNIST_train, test as MNIST_test
from models.geometric_models.mutag_gin import train as TU_train, test as TU_test
from models.geometric_models.proteins_diff_pool import Net as SAGE, train as SAGE_train, test as SAGE_test
from parser import Parser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model
def construct_model(args, dataset):
    if args.type == "SAGE":
        model = SAGE(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


def SAGE_load_data():
    print("=======Start to load data SAGE!=======")
    max_nodes = 150

    # class MyFilter(object):
    #     def __call__(self, data):
    #         return data.num_nodes <= max_nodes

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS_dense')
    dataset = TUDataset(path, name='PROTEINS', transform=T.ToDense(max_nodes))
    dataset = dataset.shuffle()
    n = len(dataset) // 10
    index = torch.randperm(len(dataset))
    train_index, val_index, test_index, retrain_index = index[:4 * n], index[4 * n:5 * n], index[
                                                                                           5 * n: 6 * n], index[
                                                                                                          6 * n:]
    # get dataset
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    retrain_dataset = dataset[retrain_index]

    # get loader
    test_loader = DenseDataLoader(test_dataset, batch_size=20)
    val_loader = DenseDataLoader(val_dataset, batch_size=20)
    train_loader = DenseDataLoader(train_dataset, batch_size=20)
    retrain_loader = DenseDataLoader(train_dataset, batch_size=20)

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
    savedpath_pretrain = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_pretrain):
        os.makedirs(savedpath_pretrain, exist_ok=True)

    if args.type == "SAGE":
        dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = SAGE_load_data()

    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    best_acc = 0
    best_val_acc = test_acc = 0

    for epoch in range(args.epochs):

        if args.type == "SAGE":
            SAGE_train(model, optimizer, train_loader, train_dataset)
            train_acc = SAGE_test(train_loader, model)
            val_acc = SAGE_test(val_loader, model)
            tmp_test_acc = SAGE_test(test_loader, model)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}')
    print(best_acc)

    # save best acc
    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)

    # save best model
    torch.save(best_model, os.path.join(savedpath_pretrain, "model.pt"))
    np.save(f"{savedpath_acc}/test_accuracy.npy", best_acc)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
