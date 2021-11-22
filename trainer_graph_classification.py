import pdb
import sys
import tqdm

from models.gin_graph_classification import Net as GIN
from models.gat_graph_classification import Net as MuGIN
# Where the program gets stuck
from models.gmt.nets import GraphMultisetTransformer
from models.gcn_graph_classification import Net as GCN
from models.gnn_cora import Net as COG
from models.pytorchtools import EarlyStopping, save_model_layer_bame
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
from parser import Parser
from models.data import get_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model

def construct_model(args, dataset, model_name):
    if args.data == "Cora" and model_name != "cog":
        print("Error: The model type for cora must be cog!")
        sys.exit(1)
    if model_name == "gin":
        model = GIN(dataset.num_features, 64, dataset.num_classes, num_layers=3)
    elif model_name == "gat":
        model = MuGIN(dataset.num_features, 32, dataset.num_classes)
    elif model_name == "gmt":
        args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
            np.mean([data.num_nodes for data in dataset]))
        model = GraphMultisetTransformer(args)
    elif model_name == "gcn":
        model = GCN(dataset.num_features, 64, dataset.num_classes)
    elif model_name == "cog":
        model = COG(dataset)
    return model


# load dataset
def loadData(args):
    model_name = args.type
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset(args.data,
                          normalize=args.normalize)
    savedpath = f"pretrained_all/pretrained_data/{model_name}_{args.data}/"
    if not os.path.isdir(savedpath):
        os.makedirs(savedpath, exist_ok=True)
    model = construct_model(args, dataset, model_name)
    if not (os.path.isfile(f"{savedpath}/train_index.pt") and os.path.isfile(
            f"{savedpath}/test_selection_index.pt") and os.path.isfile(f"{savedpath}/test_index.pt")):
        train_size = int(len(dataset) * 0.8)
        testselection_size = int(len(dataset) * 0.1)
        test_size = len(dataset) - train_size - testselection_size
        index = torch.randperm(len(dataset))
        train_dataset_index, test_selection_index, test_dataset_index = index[:train_size], index[
                                                                                            train_size:train_size + testselection_size], index[
                                                                                                                                         train_size + testselection_size:]
        os.makedirs(savedpath, exist_ok=True)
        torch.save(train_dataset_index, f"{savedpath}/train_index.pt")
        torch.save(test_selection_index, f"{savedpath}/test_selection_index.pt")
        torch.save(test_dataset_index, f"{savedpath}/test_index.pt")
    else:
        train_dataset_index = torch.load(f"{savedpath}/train_index.pt")
        test_selection_index = torch.load(f"{savedpath}/test_selection_index.pt")
        test_dataset_index = torch.load(f"{savedpath}/test_index.pt")
    train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[
        test_selection_index], dataset[test_dataset_index]

    if args.data != "Cora":
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       model_path=f"{savedpath}/model_{args.exp}.pt")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

    model = model.to(device)
    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * len(train_loader),
                                                    args.num_epochs * len(train_loader))
    return model, dataset, train_loader, test_loader, train_dataset, test_dataset, train_size, test_size, early_stopping


@torch.no_grad()
def test(loader, model):
    model.eval()
    total_correct = 0
    val_loss = 0

    for data in tqdm.tqdm(loader):
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        val_loss += loss.item() * data.num_graphs
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y).sum().item()

    return total_correct / len(loader.dataset), val_loss / len(loader.dataset)


@torch.no_grad()
def testCora(dataset, model):
    model.eval()
    dataT = dataset[0]
    log_probs, test_acc = model(), []
    for _, mask in dataT('test_mask'):
        pred = log_probs[mask].max(1)[1]
        test_acc = pred.eq(dataT.y[mask]).sum().item() / mask.sum().item()
    return test_acc


def trainCora(dataset, model, optimizer):
    dataT = dataset[0]
    model.train()
    optimizer.zero_grad()
    total_loss = F.nll_loss(model()[dataT.train_mask], dataT.y[dataT.train_mask])
    total_loss.backward()
    optimizer.step()
    return total_loss


# train model
def train(model, train_loader, train_dataset, lr_schedule, train_size, savedpath):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    if lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * train_size,
                                                    args.num_epochs * train_size)
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        if args.lr_schedule:
            scheduler.step()
    return total_loss / len(train_dataset)


# get accuracy of the model
def get_accuracy(model, test_loader, savedpath_acc):
    # create savedpath for retrained accuracy

    test_acc, test_loss = test(test_loader, model)

    with open(f"{savedpath_acc}/test_accuracy.txt", 'a') as f:
        f.write(str(test_acc) + "\n")


# train and save model -- main class
def train_and_save(args):
    # get parameters
    model_name = args.type
    num_epochs = args.epochs

    # get model and dataset
    model, dataset, train_loader, test_loader, train_dataset, test_dataset, train_size, test_size, early_stopping = loadData(args)
    originalModel = model

    # train the baseline model
    for exp in range(args.exp):
        model = originalModel
        savedpath_train = f"pretrained_all/pretrained_model/{model_name}_{args.data}_{exp}/"
        if not os.path.isdir(savedpath_train):
            os.makedirs(savedpath_train, exist_ok=True)
        for epoch in range(num_epochs):
            if args.data != "Cora":
                # train model
                train(model, train_loader, train_dataset, args.lr_schedule, train_size, savedpath_train)
                # test model
                test(test_loader, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if args.data == "Cora":
                trainCora()
                testCora(dataset)

        torch.save(model.state_dict(), os.path.join(savedpath_train, "model.pt"))

        savedpath_acc = f"pretrained_all/train_accuracy/{model_name}_{args.data}_{exp}/"
        if not os.path.isdir(savedpath_acc):
            os.makedirs(savedpath_acc, exist_ok=True)

        get_accuracy(model, test_loader, savedpath_acc)


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
