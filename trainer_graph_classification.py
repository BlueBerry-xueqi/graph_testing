import copy
import pdb

from torch_geometric.loader import DenseDataLoader
from torch_geometric.datasets import TUDataset, Planetoid
import os
import torch
import numpy as np
from parser import Parser
import os.path as osp
from models.geometric_models.GCN_model import Net as GCN
from models.geometric_models.GAT import Net as GAT
from models.geometric_models.AGNN import Net as AGNN
from models.geometric_models.ARMA import Net as ARMA
from models.geometric_models.mnist_nn_conv import Net as NN, train as MNIST_train, test as MNIST_test
import torch.nn.functional as F
from models.geometric_models.mutag_gin import Net as GIN_TU, train as TU_train, test as TU_test
from models.geomwetric_models.GraphNN import Net as GraphNN, train as GrapNN_train, test as GraphNN_test
from models.geometric_models.proteins_diff_pool import Net as SAGE, train as SAGE_train, test as SAGE_test
from models.geometric_models.mem_pool import Net as mem_pool, train as mem_train, test as mem_test
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# construct the original model
def construct_model(args, dataset):
    if args.type == "GCN":
        model = GCN(dataset.num_features, dataset.num_classes)

        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)

    elif args.type == "GAT":
        model = GAT(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    elif args.type == "AGNN":
        model = AGNN(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    elif args.type == "ARMA":
        model = ARMA(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    elif args.type == "GIN":
        model = GIN_TU(dataset.num_features, 32, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    elif args.type == "NN":
        model = NN(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    elif args.type == "GraphNN":
        model = GraphNN(dataset.num_features, dataset.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    elif args.type == "GMT":
        model = GMT(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    elif args.type == "SAGE":
        model = SAGE(dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    elif args.type == "mempool":
        model = mem_pool(dataset.num_features, 32, dataset.num_classes, dropout=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-5)

    return model, optimizer


# load dataset
def Cora_load_Data(args):
    print("=======Start to load data Cora!=======")
    dataset = 'Cora'
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = data.to(device)

    # split data : train:val:test;retrain = 4:1:1:4
    n = int(len(data.y) / 10)
    index = torch.randperm(len(data.y))
    train_index, val_index, test_index, retrain_index = index[:4 * n], index[4 * n:5 * n], index[5 * n: 6 * n], index[
                                                                                                                6 * n:]

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
    print("=======Load Cora successfully!!=======")
    return data, dataset, train_index, val_index, test_index, retrain_index


def load_data(args):
    print("=======Start to load data not CORA and SAGE!=======")
    if args.data == "MNIST":
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MNIST')
        transform = T.Cartesian(cat=False)
        dataset = MNISTSuperpixels(path, True, transform=transform)
        dataset = dataset[5000:10000]
        dataset = dataset.shuffle()
        print("load data MNIST successfully!")
    # normal TU dataset
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    retrain_loader = DataLoader(retrain_dataset, batch_size=128, shuffle=False)

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
    print("=======load data not CORA and SAGE successfully!=======")

    return dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


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
    print("=======Start to load data SAGE successfully!=======")
    return dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


def train_and_save(args):
    print("=======Start to trian ans save!=======")
    savedpath_pretrain = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_pretrain):
        os.makedirs(savedpath_pretrain, exist_ok=True)

    if args.data == "Cora":
        data, dataset, train_index, val_index, test_index, retrain_index = Cora_load_Data(args)
    if args.type == "SAGE":
        dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = SAGE_load_data()
    if args.data != "Cora" and args.type != "SAGE":
        dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = load_data(
            args)

    model, optimizer = construct_model(args, dataset)
    model = model.to(device)

    best_acc = 0
    best_val_acc = test_acc = 0

    for epoch in range(args.epochs):
        if args.data == "Cora":
            Cora_train(model, optimizer, data, train_index)
            train_acc, val_acc, test_acc = Cora_test(model, data, train_index, val_index, test_index)
        elif args.data == "MNIST":
            MNIST_train(epoch, train_loader, model, optimizer)
            train_acc = MNIST_test(train_loader, train_dataset, model)
            val_acc = MNIST_test(val_loader, val_dataset, model)
            test_acc = MNIST_test(test_loader, test_dataset, model)
        if args.type == "GraphNN":
            GrapNN_train(model, optimizer, train_loader, train_dataset)
            train_acc = GraphNN_test(train_loader, model)
            val_acc = GraphNN_test(val_loader, model)
            test_acc = GraphNN_test(test_loader, model)

        if args.type == "GIN":
            TU_train(model, train_loader, optimizer)
            train_acc = TU_test(train_loader, model)
            val_acc = TU_test(val_loader, model)
            test_acc = TU_test(test_loader, model)

        if args.type == "GMT":
            GMT_train(model, train_loader, train_dataset, optimizer)
            train_acc = GMT_test(train_loader, model)
            val_acc = GMT_test(val_loader, model)
            tmp_test_acc = GMT_test(test_loader, model)

        if args.type == "SAGE":
            SAGE_train(model, optimizer, train_loader, train_dataset)
            train_acc = SAGE_test(train_loader, model)
            val_acc = SAGE_test(val_loader, model)
            test_acc = SAGE_test(test_loader, model)

        if args.type == "mempool":
            mem_train(train_loader, model, optimizer)
            train_acc = mem_test(train_loader, model)
            # val_acc = mem_test(val_loader, model)
            test_acc = mem_test(test_loader, model)

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     test_acc = tmp_test_acc

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


def Cora_train(model, optimizer, data, train_index):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[train_index], data.y[train_index]).backward()
    optimizer.step()


@torch.no_grad()
def Cora_test(model, data, train_index, val_index, test_index):
    list_index = [train_index, val_index, test_index]
    model.eval()
    logits, accs = model(data), []
    for index in list_index:
        pred = logits[index].max(1)[1]
        acc = pred.eq(data.y[index]).sum().item() / len(index)
        accs.append(acc)
    return accs


if __name__ == "__main__":
    args = Parser().parse()
    train_and_save(args)
