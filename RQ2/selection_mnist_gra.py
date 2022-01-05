import copy
import math
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models.geometric_models.mnist_nn_conv import train as MNIST_train, test as MNIST_test
from parser import Parser
from trainer_mnist_gra import construct_model

from test_metrics.MNIST_metrics.M_Margin import margin_metrics
from test_metrics.MNIST_metrics.M_random import random_select
from test_metrics.MNIST_metrics.M_Entropy import entropy_metrics
from test_metrics.MNIST_metrics.M_DeepGini import DeepGini_metrics
from test_metrics.MNIST_metrics.M_least_confidence import least_confidence_metrics

from test_metrics.MNIST_metrics.GMM import GMM_metrics
from test_metrics.MNIST_metrics.Agg import AgglomerativeClustering_metrics
from test_metrics.MNIST_metrics.Kmeans_selection import KMeans_score
from test_metrics.MNIST_metrics.MCP import mcp_score
from test_metrics.MNIST_metrics.Spectral import SpectralClustering_metrics
from test_metrics.MNIST_metrics.variance import computeVariancescore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadData():
    savedpath_index = f"pretrained_all/pretrained_model/{args.type}_{args.data}/index"

    if not (os.path.isfile(f"{savedpath_index}/train_index.pt") and os.path.isfile(
            f"{savedpath_index}/val_index.pt") and os.path.isfile(
        f"{savedpath_index}/test_index.pt")):
        print("No prepared index")
    else:
        dataset = torch.load(f"{savedpath_index}/dataset.pt")
        train_index = torch.load(f"{savedpath_index}/train_index.pt")
        val_index = torch.load(f"{savedpath_index}/val_index.pt")
        test_index = torch.load(f"{savedpath_index}/test_index.pt")
        # get dataset
        train_dataset = dataset[train_index]
        val_dataset = dataset[val_index]
        test_dataset = dataset[test_index]

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return dataset, train_index, test_index, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def retrain_and_save(args):
    dataset, train_index, test_index, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = loadData()
    mse_list = []

    # do three exps
    for exp_ID in range(100):
        model, optimizer = construct_model(args, dataset)
        model = model.to(device)

        savedpath_model = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
        model.load_state_dict(torch.load(os.path.join(savedpath_model, "model.pt"), map_location=device))

        ratio = args.select_ratio
        len_test = len(test_index)
        select_num = int(ratio / 100 * len_test)

        # get select index using metrics
        select_index = select_functions(model, len_test, args.metrics, test_loader, select_num, dataset)
        sample_index = test_index[select_index]
        sample_dataset = dataset[sample_index]
        sample_loader = DataLoader(sample_dataset, batch_size=64, shuffle=True)
        sample_test_acc = MNIST_test(sample_loader, sample_dataset, model)

        savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/test_accuracy.npy"
        test_acc = np.load(savedpath_acc)
        diff = test_acc - sample_test_acc
        mse_power = np.power(diff, 2)
        mse_list.append(mse_power)

        print(f'Exp: {exp_ID:03d},  Test acc: {test_acc:.4f}, sample acc: {sample_test_acc:.4f}, '
              f'difference: {diff:.4f}')

    mes_avg = math.sqrt(np.sum(mse_list) / 100)
    print("mse is: ", mes_avg)

    # save best accuracy
    savedpath_avg = f"MSE/{args.type}_{args.data}/{args.metrics}/"
    if not os.path.isdir(savedpath_avg):
        os.makedirs(savedpath_avg, exist_ok=True)
    np.save(f"{savedpath_avg}/mse{ratio}.npy", mes_avg)


def select_functions(model, retrain_dataset_length, metric, retrain_loader, select_num, dataset):
    if metric == "deepgini":
        selected_index = DeepGini_metrics(model, retrain_loader, select_num)
    elif metric == "random":
        selected_index = random_select(retrain_dataset_length, select_num)
    elif metric == "l_con":
        selected_index = least_confidence_metrics(model, select_num, retrain_loader)
    elif metric == "entropy":
        selected_index = entropy_metrics(model, retrain_loader, select_num)
    elif metric == "margin":
        selected_index = margin_metrics(model, retrain_loader, select_num)

    elif metric == "MCP":
        selected_index = mcp_score(model, retrain_loader, select_num, dataset.num_classes, 64)
    elif metric == "variance":
        selected_index = computeVariancescore(model, retrain_loader, select_num)
    elif metric == "kmeans":
        selected_index = KMeans_score(model, retrain_loader, select_num)
    elif metric == "spec":
        selected_index = SpectralClustering_metrics(model, retrain_loader, select_num)
    elif metric == "Hierarchical":
        selected_index = AgglomerativeClustering_metrics(model, retrain_loader, select_num)
    elif metric == "GMM":
        selected_index = GMM_metrics(model, retrain_loader, select_num)

    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
