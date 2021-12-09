import copy
from models.GNN_models import Net as GNN
from models.gin_graph_classification import Net as GIN
from models.gat_graph_classification import Net as GAT
from models.gmt.nets import GraphMultisetTransformer as GMT
from models.gcn_graph_classification import GCNConv as GCN
from models.Tudataset_models.gin_network import GIN as TUGIN
from models.Tudataset_models.gnn_architectures import GINWithJK as GINJFK, GINE0, GINE, GINEWithJK

from models.pytorchtools import EarlyStopping, save_model_layer_bame
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import os
import torch
import torch.nn.functional as F
import numpy as np
from parser import Parser

import os.path as osp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model

def construct_model(args, dataset, train_loader, model_path):
    if args.type == "gin":
        model = GIN(dataset.num_features, 64, dataset.num_classes, num_layers=3)
    elif args.type == "gat":
        model = GAT(dataset.num_features, 32, dataset.num_classes)
    elif args.type == "gmt":
        args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
            np.mean([data.num_nodes for data in dataset]))
        model = GMT(args)
    elif args.type == "gcn":
        model = GCN(dataset.num_features, dataset.num_classes)
    elif args.type == "gnn":
        model = GNN(dataset)
    elif args.type == "TUGIN":
        model = TUGIN(dataset, 5, 64)
    elif args.type == "GINJFK":
        model = GINJFK(dataset, 5, 64)
    elif args.type == "GINE":
        model = GINE(dataset, 3, 64)
    elif args.type == "GINEJFK":
        model = GINEWithJK(dataset, 3, 64)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, model_path=f"{model_path}/model.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # if args.lr_schedule:
    #     scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * len(train_loader),
    #                                                 args.num_epochs * len(train_loader))
    # else:
    #     scheduler = None

    return model, early_stopping, optimizer


# load dataset
def loadData(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.data)
    dataset = TUDataset(path, name=args.data)
    dataset = dataset.shuffle()
    savedpath = f"pretrained_all/pretrained_data/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath):
        os.makedirs(savedpath, exist_ok=True)
    n = len(dataset) // 10
    train_dataset = dataset[:n * 4]
    retrain_dataset = dataset[n * 4:n * 8]
    test_dataset = dataset[n * 8:]

    torch.save(dataset, f"{savedpath}/dataset.pt")
    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)
    retrain_loader = DataLoader(retrain_dataset, batch_size=60, shuffle=False)

    return dataset, train_loader, retrain_loader, test_loader, train_dataset, retrain_dataset, test_dataset


def train(model, train_loader, optimizer, train_dataset):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader, model):
    model.eval()
    correct = 0
    val_loss = 0
    for data in loader:
        data = data.to(device)

        out = model(data)
        loss = F.nll_loss(out, data.y)
        val_loss += loss.item() * data.num_graphs

        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), val_loss / len(loader.dataset)


# train and save model -- main class
def train_and_save(args):
    # get parameters
    model_name = args.type
    num_epochs = args.epochs

    # get dataset
    dataset, train_loader, retrain_loader, test_loader, train_dataset, retrain_dataset, test_dataset = loadData(args)
    # train the baseline model
    best_acc = 0
    savedpath_pretrain = f"pretrained_all/pretrained_model/{model_name}_{args.data}/"
    if not os.path.isdir(savedpath_pretrain):
        os.makedirs(savedpath_pretrain, exist_ok=True)
    # get initial model
    model, early_stopping, optimizer = construct_model(args, dataset, train_loader, savedpath_pretrain)
    model = model.to(device)
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, train_dataset)
        # if args.lr_schedule:
        #     scheduler.step()

        train_acc, train_loss = test(train_loader, model)
        test_acc, test_loss = test(test_loader, model)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

        # early_stopping(test_loss, model,
        #                performance={"train_acc": train_acc, "test_acc": test_acc, "train_loss": train_loss,
        #                             "test_loss": test_loss})
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
              f'Test Acc: {test_acc:.5f}')

    savedpath_acc = f"pretrained_all/train_accuracy/{model_name}_{args.data}/"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    print("Best acc", best_acc)
    torch.save(best_model, os.path.join(savedpath_pretrain, "model.pt"))
    np.save(f"{savedpath_acc}/test_accuracy.npy", best_acc)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
