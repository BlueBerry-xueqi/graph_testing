import copy
import pdb
import sys

import numpy

from test_metrics.Entropy import entropy
from test_metrics.Margin import margin_score
from test_metrics.max_probability import max_score
from trainer_graph_classification import construct_model, test
import os
import torch
import numpy as np
from parser import Parser
from test_metrics.DeepGini import deepgini_score
from test_metrics.random import random_select
from trainer_graph_classification import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadData(args, test_index=None):
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
    data, dataset, train_index, val_index, test_index, retrain_index = loadData(args)

    len_retrain_set = len(retrain_index)
    test_acc_exp = 0

    # do three exps
    for exp_ID in range(args.exp):
        print("exp", exp_ID, "...")
        # set savedpath
        savedpath_retrain = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/"
        if not os.path.isdir(savedpath_retrain):
            os.makedirs(savedpath_retrain, exist_ok=True)

        # construct original model
        model, optimizer = construct_model(args, dataset, savedpath_retrain)
        model = model.to(device)

        # get select ratio
        ratio = args.select_ratio
        # get the previous ratio
        ratio_pre = ratio - 10

        # the first retrain use the pretrained model
        if ratio == 10:
            savedpath_pre = f"pretrained_all/pretrained_model/{args.type}_{args.data}/model.pt"
            if os.path.isfile(savedpath_pre):
                model.load_state_dict(torch.load(savedpath_pre, map_location=device))
                print("pretrained model load successful!!")
            else:
                print("pre-trained model not exists")
                sys.exit()

        # for other's use the previous retrained model
        if not ratio == 10:
            savedpath_previous = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/model{ratio_pre}.pt"
            # load model from previous model
            if os.path.isfile(savedpath_previous):
                model.load_state_dict(torch.load(savedpath_previous, map_location=device))
                print("retrained model load successful!!")
            else:
                print("retrained model not exists")
                sys.exit()

        # save selected index
        savedpath_index = f"retrained_all/selected_index/{args.metrics}/{args.type}_{args.data}/"
        if not os.path.isdir(savedpath_index):
            os.makedirs(savedpath_index, exist_ok=True)
            # get retrain loader for select functions
            all_index = select_functions(model, retrain_index, len_retrain_set,
                                         args.metrics, data)
            torch.save(all_index, os.path.join(savedpath_index, "all_index.pt"))
        else:
            # if the indexes are already saved, use the index directly.
            all_index = torch.load(f"{savedpath_index}all_index.pt")
            print("sorted indexed loaded")

        # take this length from the retrain set
        length_current = int(ratio / 100 * len_retrain_set)
        length_pre = int(ratio_pre / 100 * len_retrain_set)

        select_index = all_index[:length_current]
        select_index = numpy.array(select_index)
        select_index = torch.from_numpy(select_index)
        new_train_index = torch.cat((select_index, train_index), -1)

        # initiallize the best accuracy
        best_acc = 0
        best_val_acc = test_acc = 0
        for epoch in range(args.retrain_epochs):
            train(model, optimizer, data, new_train_index)

            train_acc, val_acc, tmp_test_acc = test(model, data, train_index, val_index, test_index)
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

    # save the best model
    torch.save(best_exp_model, os.path.join(savedpath_retrain, f"model{ratio}.pt"))

    # save the best accuracy
    savedpath_acc = f"retrained_all/train_accuracy/{args.metrics}/{args.type}_{args.data}"
    if not os.path.isdir(savedpath_acc):
        os.makedirs(savedpath_acc, exist_ok=True)
    np.save(f"{savedpath_acc}/test_accuracy_ratio{ratio}.npy", test_acc_exp)


def select_functions(model, retrain_index, select_num, metric, data):
    if metric == "DeepGini":
        selected_index = deepgini_score(model, retrain_index, select_num, data)
    elif metric == "random":
        selected_index = random_select(len(retrain_index), select_num)
    elif metric == "max_probability":
        selected_index = max_score(model, retrain_index, select_num, data)
    elif metric == "Entropy":
        selected_index = entropy(model, retrain_index, select_num, data)
    if metric == "margin":
        selected_index = margin_score(model, retrain_index, select_num, data)
    # elif metric == "dsa":
    #     selected_index = fetch_dsa(model, train_loader, target_loader, "test_selection_loader", [], select_num)
    # elif metric == "lsa":
    #     selected_index = fetch_lsa(model, target_loader, select_num)

    # elif metric == "MCP":
    #     selected_index = MCP_score(model, target_loader, select_num, ncl=10)
    # elif metric == "CES":
    #     selected_index = conditional_sample(model, target_loader, select_num, attack=0)
    # # if metric == "ModelWrapper":
    # #     scores, _, _ = deepgini_score(model, target_loader)
    # if metric == "nc":
    #     scores, _, _ = nbc
    # if metric == "sa":
    #     scores, _, _ = deepgini_score(model, target_loader)
    # if metric == "sihoutete":
    #     scores, _, _ = deepgini_score(model, target_loader)

    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
