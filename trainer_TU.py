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

    return dataset, train_loader, val_loader, test_loader, retrain_loader, train_dataset,val_dataset, test_dataset, retrain_dataset


def train_and_save(args):
    # load data
    dataset, train_loader, val_loader, test_loader, retrain_loader, train_dataset,val_dataset, test_dataset, retrain_dataset = load_data(
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
        val_acc = GraphNN_test(val_loader, model)

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
    test_acc = GraphNN_test(test_loader, model)
    print("best test accuracy is: ", test_acc)

    # save best accuracy
    savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy.npy", test_acc)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
