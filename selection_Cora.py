import copy
import os
import sys

import numpy as np
import torch

from parser import Parser
from test_metrics.Cora_metrics.Cora_DeepGini import DeepGini_metrics
from test_metrics.Cora_metrics.Cora_random import random_select
from test_metrics.Cora_metrics.Cora_BALD import BALD_metrics
from test_metrics.Cora_metrics.Cora_Margin import margin_metrics
from test_metrics.Cora_metrics.Cora_Kmeans_selection import Kmeans_metrics
from test_metrics.Cora_metrics.Cora_Entropy import entropy_metrics
from test_metrics.Cora_metrics.Cora_least_confidence import least_confidence_metrics

from trainer_Cora import construct_model, Cora_test, Cora_train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Cora_loadData(args):
    savedpath_index = f"pretrained_all/pretrained_model/{args.type}_{args.data}/index"

    if not (os.path.isfile(f"{savedpath_index}/train_index.pt") and os.path.isfile(
            f"{savedpath_index}/val_index.pt") and os.path.isfile(
        f"{savedpath_index}/test_index.pt") and os.path.isfile(f"{savedpath_index}/retrain_index.pt")):
        print("No prepared index")
    else:
        dataset = torch.load(f"{savedpath_index}/dataset.pt")
        data = dataset[0]
        data = data.to(device)
        train_index = torch.load(f"{savedpath_index}/train_index.pt")
        val_index = torch.load(f"{savedpath_index}/val_index.pt")
        test_index = torch.load(f"{savedpath_index}/test_index.pt")
        retrain_index = torch.load(f"{savedpath_index}/retrain_index.pt")

    return data, dataset, train_index, val_index, test_index, retrain_index


def retrain_and_save(args):
    data, dataset, train_index, val_index, test_index, retrain_index = Cora_loadData(args)
    test_acc_exp = 0

    for exp_ID in range(args.exp):
        print("exp", exp_ID, "...")

        # construct original model
        model, optimizer = construct_model(args, dataset)
        model = model.to(device)

        ratio = args.select_ratio
        ratio_pre = ratio - 5

        # the first retrain use the pretrained model
        if ratio == 5:
            savedpath_pre = f"pretrained_all/pretrained_model/{args.type}_{args.data}/model.pt"
            if os.path.isfile(savedpath_pre):
                model.load_state_dict(torch.load(savedpath_pre, map_location=device))
                print("pretrained model load successful!!")
            else:
                print("pre-trained model not exists")
                sys.exit()

        # for other's use the previous retrained model
        if not ratio == 5:
            savedpath_previous = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/model{ratio_pre}.pt"
            # load model from previous model
            if os.path.isfile(savedpath_previous):
                model.load_state_dict(torch.load(savedpath_previous, map_location=device))
                print("retrained model load successful!!")
            else:
                print("retrained model not exists")
                sys.exit()

        retrain_length = len(retrain_index)
        select_num = int(ratio / 100 * retrain_length)

        select_index = Cora_select_functions(model, retrain_index, args.metrics, data, select_num)
        new_select_index = retrain_index[select_index]
        new_train_index = torch.cat((new_select_index, train_index), -1)


        best_acc = 0
        best_val_acc = test_acc = 0
        for epoch in range(args.retrain_epochs):
            Cora_train(model, optimizer, data, new_train_index)
            train_acc, val_acc, tmp_test_acc = Cora_test(model, data, new_train_index, val_index, test_index)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc

            if test_acc > best_acc:
                best_model = copy.deepcopy(model.state_dict())
                best_acc = test_acc

            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

        print("exp ID: ", exp_ID, ", best accuracy is：", best_acc)

        if best_acc > test_acc_exp:
            print("accuracy increase!")
            best_exp_model = copy.deepcopy(best_model)
            test_acc_exp = best_acc

    print("final best accuracy is：", test_acc_exp)

    # save best model
    savedpath_model = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_model):
        os.makedirs(savedpath_model, exist_ok=True)
    torch.save(best_exp_model, os.path.join(savedpath_model, f"model{ratio}.pt"))

    # save best accuracy
    savedpath_acc = f"retrained_all/train_accuracy/{args.metrics}/{args.type}_{args.data}"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy_ratio{ratio}.npy", test_acc_exp)


def Cora_select_functions(model, retrain_index, metric, data, select_num):
    if metric == "deepgini":
        selected_index = DeepGini_metrics(model, retrain_index, data, select_num)
    elif metric == "random":
        selected_index = random_select(len(retrain_index), select_num)
    elif metric == "l_con":
        selected_index = least_confidence_metrics(model, retrain_index, data, select_num)
    elif metric == "entropy":
        selected_index = entropy_metrics(model, retrain_index, data, select_num)
    elif metric == "margin":
        selected_index = margin_metrics(model, retrain_index, data, select_num)
    elif metric == "kmeans":
        selected_index = Kmeans_metrics(model, retrain_index, data, select_num)
    elif metric == "bald":
        selected_index = BALD_metrics(model, retrain_index, data, select_num)
    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
