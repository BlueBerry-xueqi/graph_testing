import copy
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DenseDataLoader


from parser import Parser
from test_metrics.diff_metrics.BALD import BALD_metrics
from test_metrics.diff_metrics.DeepGini import DeepGini_metrics
from test_metrics.diff_metrics.Entropy import entropy_metrics
from test_metrics.diff_metrics.Kmeans_selection import Kmeans_metrics
from test_metrics.diff_metrics.Margin import margin_metrics
from test_metrics.diff_metrics.least_confidence import least_confidence_metrics
from test_metrics.diff_metrics.random import random_select
from models.geometric_models.proteins_diff_pool import train as diff_train, test as diff_test
from trainer_diff import construct_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadData():
    savedpath_index = f"pretrained_all/pretrained_model/{args.type}_{args.data}/index"

    if not (os.path.isfile(f"{savedpath_index}/train_index.pt") and os.path.isfile(
            f"{savedpath_index}/val_index.pt") and os.path.isfile(
        f"{savedpath_index}/test_index.pt") and os.path.isfile(f"{savedpath_index}/retrain_index.pt")):
        print("No prepared index")
    else:
        dataset = torch.load(f"{savedpath_index}/dataset.pt")
        train_index = torch.load(f"{savedpath_index}/train_index.pt")
        val_index = torch.load(f"{savedpath_index}/val_index.pt")
        test_index = torch.load(f"{savedpath_index}/test_index.pt")
        retrain_index = torch.load(f"{savedpath_index}/retrain_index.pt")
        # get dataset
        train_dataset = dataset[train_index]
        val_dataset = dataset[val_index]
        test_dataset = dataset[test_index]
        retrain_dataset = dataset[retrain_index]

        train_loader = DenseDataLoader(train_dataset, batch_size=60, shuffle=True)
        val_loader = DenseDataLoader(val_dataset, batch_size=60, shuffle=False)
        test_loader = DenseDataLoader(test_dataset, batch_size=60, shuffle=False)
        retrain_loader = DenseDataLoader(retrain_dataset, batch_size=60, shuffle=False)

    return dataset, train_index, val_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


def retrain_and_save(args):
    dataset, train_index, val_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = loadData()
    best_exp_acc = 0
    # do three experiments
    for exp_ID in range(args.exp):
        print("exp", exp_ID, "...")
        # construct model
        model, optimizer = construct_model(args, dataset)
        model = model.to(device)

        # get ratio budget
        ratio = args.select_ratio

        # the first retrain with pretrained model
        if ratio == 5:
            savedpath_pretrained = f"pretrained_all/pretrained_model/{args.type}_{args.data}/model.pt"
            if os.path.isfile(savedpath_pretrained):
                model.load_state_dict(torch.load(savedpath_pretrained, map_location=device))
                print("pretrained model load successful!!")
            else:
                print("pre-trained model not exists")
                sys.exit()

        # the second and following retrain use the previous retrained model
        if not ratio == 5:
            p_ratio = ratio - 5
            savedpath_previous = f"retrained_all/retrained_model/{args.type}_{args.data}/{args.metrics}/model{p_ratio}.pt"
            if os.path.isfile(savedpath_previous):
                model.load_state_dict(torch.load(savedpath_previous, map_location=device))
                print("retrained model load successful!!")
            else:
                print("retrained model not exists")
                sys.exit()

        len_retrain_set = len(retrain_dataset)
        select_num = int(ratio / 100 * len_retrain_set)

        # get select index using metrics
        select_index = select_functions(model, len_retrain_set, args.metrics, retrain_loader, select_num)
        new_select_index = retrain_index[select_index]
        new_train_index = np.concatenate((new_select_index, train_index))
        new_train_dataset = dataset[new_train_index]
        new_train_loader = DenseDataLoader(new_train_dataset, batch_size=60, shuffle=True)

        best_acc = 0
        for epoch in range(args.retrain_epochs):
            diff_train(model, optimizer, new_train_loader, new_train_dataset)
            train_acc = diff_test(model, train_loader)
            val_acc = diff_test(model, val_loader)

            # select best test accuracy and save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())

            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}')
        print("best val acc is: ", best_acc)
        # if test acc is better than the best exp acc, copy the best exp model
        if best_acc > best_exp_acc:
            best_exp_model = copy.deepcopy(best_model)
            best_exp_acc = best_acc

    print("choose model with best val: ", best_exp_acc)
    # load best model and get test acc
    model.load_state_dict(best_exp_model)
    test_acc = diff_test(model, test_loader)
    print("best test accuracy is: ", test_acc)

    # save best model
    savedpath_model = f"retrained_all/retrained_model/{args.type}_{args.data}/{args.metrics}/"
    if not os.path.isdir(savedpath_model):
        os.makedirs(savedpath_model, exist_ok=True)
    torch.save(best_exp_model, os.path.join(savedpath_model, f"model{ratio}.pt"))

    # save the best accuracy
    savedpath_acc = f"retrained_all/train_accuracy/{args.type}_{args.data}/{args.metrics}"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy_ratio{ratio}.npy", test_acc)


def select_functions(model, retrain_dataset_length, metric, retrain_loader, select_num):
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
    elif metric == "kmeans":
        selected_index = Kmeans_metrics(model, retrain_loader, select_num)
    elif metric == "bald":
        selected_index = BALD_metrics(model, retrain_loader, select_num, retrain_dataset_length)

    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
