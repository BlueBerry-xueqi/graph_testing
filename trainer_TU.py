import copy
from torch_geometric.datasets import TUDataset, Planetoid
import os
import torch
import numpy as np
from parser import Parser
import os.path as osp
from models.geometric_models.GraphNN import Net as GraphNN, train as GrapNN_train, test as GraphNN_test
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model: GraphConv, dataset: NCI109, Tox21_AhR_training, NCI1

# construct the original model
def construct_model(args, dataset):
    if args.type == "GraphNN":
        model = GraphNN(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    return model, optimizer


def load_data(args):
    # load data from TU dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'TU')
    dataset = TUDataset(path, name=args.data).shuffle()

    # split dataset
    n = len(dataset) // 10
    index = torch.randperm(len(dataset))
    train_index, test_index, retrain_index = index[:4 * n], index[4 * n: 6 * n], index[6 * n:]
    # get dataset
    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]
    retrain_dataset = dataset[retrain_index]

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    retrain_loader = DataLoader(retrain_dataset, batch_size=128, shuffle=False)

    # save index path
    savedpath_dataset = f"pretrained_all/pretrained_model/{args.type}_{args.data}/dataset"
    if not os.path.isdir(savedpath_dataset):
        os.makedirs(savedpath_dataset, exist_ok=True)

    # save dataset
    torch.save(dataset, f"{savedpath_dataset}/dataset.pt")
    torch.save(train_index, f"{savedpath_dataset}/train_index.pt")
    torch.save(test_index, f"{savedpath_dataset}/test_index.pt")
    torch.save(retrain_index, f"{savedpath_dataset}/retrain_index.pt")

    return dataset, train_loader, test_loader, retrain_loader, train_dataset, test_dataset, retrain_dataset


def train_and_save(args):
    # load data
    dataset, train_loader, test_loader, retrain_loader, train_dataset, test_dataset, retrain_dataset = load_data(
        args)
    # construct model
    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    # train model in many epochs
    best_acc = 0
    for epoch in range(args.epochs):
        # train model
        GrapNN_train(model, optimizer, train_loader, train_dataset)
        # get test accuracy of model
        train_acc = GraphNN_test(train_loader, model)
        test_acc = GraphNN_test(test_loader, model)

        # select best test accuracy and save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
              f'Test: {test_acc:.4f}')
    print("Best acuracy is: ", best_acc)

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


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
