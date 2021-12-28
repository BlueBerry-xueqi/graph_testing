import copy
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from parser import Parser
from models.geometric_models.mem_pool import train as mem_train, test as mem_test
from trainer_mempool import construct_model
from test_metrics.mem_metrics.mem_Margin import margin_metrics
from test_metrics.mem_metrics.mem_BALD import BALD_metrics
from test_metrics.mem_metrics.mem_random import random_select
from test_metrics.mem_metrics.mem_Entropy import entropy_metrics
from test_metrics.mem_metrics.mem_DeepGini import DeepGini_metrics
from test_metrics.mem_metrics.mem_Kmeans_selection import Kmeans_metrics
from test_metrics.mem_metrics.mem_least_confidence import least_confidence_metrics

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

        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
        retrain_loader = DataLoader(retrain_dataset, batch_size=20, shuffle=False)

    return dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


def retrain_and_save(args):
    dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = loadData()
    test_acc_exp = 0
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
            savedpath_previous = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/model{ratio - 5}.pt"
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
        new_train_dataset = new_train_dataset.shuffle()
        new_train_loader = DataLoader(new_train_dataset, batch_size=20, shuffle=True)

        best_acc = best_val_acc = test_acc = 0
        for epoch in range(args.retrain_epochs):
            mem_train(new_train_loader, model, optimizer)
            train_acc = mem_test(new_train_loader, model)
            val_acc = mem_test(val_loader, model)
            if test_acc > best_acc:
                best_model = copy.deepcopy(model.state_dict())
                best_acc = test_acc
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                test_acc = mem_test(test_loader, model)

            print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
                  f'Test: {test_acc:.4f}')
        print("exp ID: ", exp_ID, ", best accuracy is：", best_acc)

        if best_acc > test_acc_exp:
            print("accuracy increase!")
            best_exp_model = copy.deepcopy(best_model)
            test_acc_exp = best_acc

    print("final best accuracy is：", test_acc_exp)

    # save the best model
    savedpath_model = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/"
    if not os.path.isdir(savedpath_model):
        os.makedirs(savedpath_model, exist_ok=True)
    torch.save(best_exp_model, os.path.join(savedpath_model, f"model{ratio}.pt"))

    # save the best accuracy
    savedpath_acc = f"retrained_all/train_accuracy/{args.metrics}/{args.type}_{args.data}"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy_ratio{ratio}.npy", test_acc_exp)


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
