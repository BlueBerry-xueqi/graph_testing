import copy
import pdb
import sys
import tqdm
from models.GNN_models import Net as GNN
from models.gin_graph_classification import Net as GIN
from models.gat_graph_classification import Net as MuGIN
from models.gmt.nets import GraphMultisetTransformer
from models.gcn_graph_classification import Net as GCN
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

def construct_model(args, dataset, train_loader, model_path):
    if args.type == "gin":
        model = GIN(dataset.num_features, 64, dataset.num_classes, num_layers=3)
    elif args.type == "gat":
        model = MuGIN(dataset.num_features, 32, dataset.num_classes)
    elif args.type == "gmt":
        args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
            np.mean([data.num_nodes for data in dataset]))
        model = GraphMultisetTransformer(args)
    elif args.type == "gcn":
        model = GCN(dataset.num_features, 64, dataset.num_classes)

    elif args.type == "gnn":
        model = GNN(dataset)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, model_path=f"{model_path}/model.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = model.to(device)
    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * len(train_loader),
                                                    args.num_epochs * len(train_loader))
    else:
        scheduler = None

    return model, early_stopping, optimizer, scheduler


# load dataset
def loadData(args):
    dataset = get_dataset(args.data, normalize=args.normalize)
    dataset = dataset.shuffle()

    savedpath = f"pretrained_all/pretrained_data/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath):
        os.makedirs(savedpath, exist_ok=True)
    if not (os.path.isfile(f"{savedpath}/train_index.pt") and os.path.isfile(
            f"{savedpath}/test_selection_index.pt") and os.path.isfile(f"{savedpath}/test_index.pt")):
        train_size = int(len(dataset) * 0.4)
        testselection_size = int(len(dataset) * 0.5)
        index = torch.randperm(len(dataset))
        train_dataset_index, test_selection_index, test_dataset_index = index[:train_size], index[
                                                                                            train_size:train_size + testselection_size], index[
                                                                                                                           train_size + testselection_size:]
        torch.save(dataset, f"{savedpath}/dataset.pt")
        torch.save(train_dataset_index, f"{savedpath}/train_index.pt")
        torch.save(test_selection_index, f"{savedpath}/test_selection_index.pt")
        torch.save(test_dataset_index, f"{savedpath}/test_index.pt")
    else:
        train_dataset_index = torch.load(f"{savedpath}/train_index.pt")
        test_selection_index = torch.load(f"{savedpath}/test_selection_index.pt")
        test_dataset_index = torch.load(f"{savedpath}/test_index.pt")
    train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[test_selection_index], \
                                                          dataset[test_dataset_index]

    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)
    test_selection_loader = DataLoader(test_selection_dataset, batch_size=60, shuffle=False)

    return dataset, train_loader, test_loader, test_selection_loader, train_dataset_index, test_selection_dataset, test_selection_index, train_dataset


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
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset), val_loss / len(loader.dataset)


# train and save model -- main class
def train_and_save(args):
    # get parameters
    model_name = args.type
    num_epochs = args.epochs

    # get dataset
    dataset, train_loader, test_loader, _, _, _, _, train_dataset = loadData(args)
    # train the baseline model
    best_acc = 0
    savedpath_pretrain = f"pretrained_all/pretrained_model/{model_name}_{args.data}/"
    if not os.path.isdir(savedpath_pretrain):
        os.makedirs(savedpath_pretrain, exist_ok=True)
    # get initial model
    model, early_stopping, optimizer, scheduler = construct_model(args, dataset, train_loader, savedpath_pretrain)
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
