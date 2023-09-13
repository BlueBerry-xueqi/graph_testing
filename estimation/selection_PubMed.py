import math
import os

import numpy as np
import torch

from parser import Parser
from test_metrics.Cora_metrics.Cora_random import random_select
from test_metrics.Cora_metrics.kmeans_plus import KMeans_plusplus_score
from trainer_Cora import construct_model, Cora_test

from test_metrics.Cora_metrics.GMM import GMM_metrics
from test_metrics.Cora_metrics.Agg import AgglomerativeClustering_metrics
from test_metrics.Cora_metrics.Kmeans_selection import KMeans_score
from test_metrics.Cora_metrics.Spectral import SpectralClustering_metrics
from test_metrics.Cora_metrics.mini_batch import MiniBatchKMeans_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Cora_loadData(args):
    savedpath_index = f"pretrained_all/pretrained_model/{args.type}_{args.data}/index"

    if not (os.path.isfile(f"{savedpath_index}/train_index.pt") and os.path.isfile(
            f"{savedpath_index}/val_index.pt") and os.path.isfile(
        f"{savedpath_index}/test_index.pt")):
        print("No prepared index")
    else:
        dataset = torch.load(f"{savedpath_index}/dataset.pt")
        data = dataset[0]
        data = data.to(device)
        train_index = torch.load(f"{savedpath_index}/train_index.pt")
        val_index = torch.load(f"{savedpath_index}/val_index.pt")
        test_index = torch.load(f"{savedpath_index}/test_index.pt")

    return data, dataset, train_index, val_index, test_index


def retrain_and_save(args):
    data, dataset, train_index, val_index, test_index = Cora_loadData(args)
    finial_mse_list = []
    for exp in range(5):
        mse_list = []
        for exp_ID in range(args.mse_epochs):
            # construct original model
            model, optimizer = construct_model(args, dataset)
            model = model.to(device)

            # load model
            savedpath_model = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
            model.load_state_dict(torch.load(os.path.join(savedpath_model, "model.pt"), map_location=device))

            # get selected index
            select_num = args.select_num
            select_index = Cora_select_functions(model, test_index, args.metrics, data, select_num)
            new_select_index = test_index[select_index]

            # get test accuracy of the small sample and the total test dataset
            sample_test_acc = Cora_test(model, data, new_select_index)
            savedpath_acc = f"pretrained_all/train_accuracy/{args.type}_{args.data}/test_accuracy.npy"
            test_acc = np.load(savedpath_acc)
            diff = test_acc - sample_test_acc
            mse_power = np.power(diff, 2)
            mse_list.append(mse_power)

            print(f'Exp: {exp_ID:03d},  Test acc: {test_acc:.4f}, sample acc: {sample_test_acc:.4f}, '
                  f'difference: {diff:.4f}')

        mes_avg = math.sqrt(np.sum(mse_list) / args.mse_epochs)
        print("mse is: ", mes_avg)
        finial_mse_list.append(mes_avg)

    mse_final = np.min(finial_mse_list)
    print("mse_final is: ", mse_final)
    # save best accuracy
    savedpath_avg = f"MSE/{args.type}_{args.data}/{args.metrics}/"
    if not os.path.isdir(savedpath_avg):
        os.makedirs(savedpath_avg, exist_ok=True)
    np.save(f"{savedpath_avg}/mse{select_num}.npy", mse_final)


def Cora_select_functions(model, retrain_index, metric, data, select_num):
    if metric == "random":
        selected_index = random_select(len(retrain_index), select_num)
    elif metric == "kmeans":
        selected_index = KMeans_score(model, retrain_index, data, select_num)
    elif metric == "spec":
        selected_index = SpectralClustering_metrics(model, retrain_index, data, select_num)
    elif metric == "Hierarchical":
        selected_index = AgglomerativeClustering_metrics(model, retrain_index, data, select_num)
    elif metric == "GMM":
        selected_index = GMM_metrics(model, retrain_index, data, select_num)
    elif metric == "mini":
        selected_index = MiniBatchKMeans_metrics(model, retrain_index, data, select_num)
    elif metric == "kplus":
        selected_index = KMeans_plusplus_score(model, retrain_index, data, select_num)

    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
