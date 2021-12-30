import copy
from torch_geometric.datasets import TUDataset, Planetoid
import os
import torch
import numpy as np
from parser import Parser
import os.path as osp
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from models.Tudataset_models.gnn_architectures import GINE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model: GraphConv, dataset: NCI109, Tox21_AhR_training, NCI1

# construct the original model
def construct_model(args, dataset):
    if args.type == "GINE":
        model = GINE(dataset, 3, 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def load_data(args):
    # load data from TU dataset
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', 'TU')
    dataset = TUDataset(path, name=args.data).shuffle()
    dataset = dataset[:5000]

    # split dataset
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
    savedpath_dataset = f"pretrained_all/pretrained_model/{args.type}_{args.data}/dataset"
    if not os.path.isdir(savedpath_dataset):
        os.makedirs(savedpath_dataset, exist_ok=True)

    # save dataset
    torch.save(dataset, f"{savedpath_dataset}/dataset.pt")
    torch.save(train_index, f"{savedpath_dataset}/train_index.pt")
    torch.save(val_index, f"{savedpath_dataset}/val_index.pt")
    torch.save(test_index, f"{savedpath_dataset}/test_index.pt")
    torch.save(retrain_index, f"{savedpath_dataset}/retrain_index.pt")

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
    best_val_acc = 0.0
    best_test = 0.0
    for epoch in range(args.epochs):
        # train model
        GINE_train(train_loader, model, optimizer)
        train_acc = GINE_test(train_loader, model)
        val_acc = GINE_test(val_loader, model)
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
    test_acc = GINE_test(test_loader, model)
    print("best test accuracy is: ", test_acc)

    # save best accuracy
    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy.npy", test_acc)


# One training epoch for GNN model.
def GINE_train(train_loader, model, optimizer):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def GINE_test(loader, model):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
