import os

import numpy as np
import torch

from parser import Parser
from test_metrics.Cora_metrics.Cora_DeepGini import DeepGini_metrics
from test_metrics.Cora_metrics.Cora_Entropy import entropy_metrics
from test_metrics.Cora_metrics.Cora_Margin import margin_metrics
from test_metrics.Cora_metrics.Cora_least_confidence import least_confidence_metrics
from test_metrics.Cora_metrics.Cora_random import random_select
from test_metrics.Cora_metrics.MCP import mcp_score
from test_metrics.Cora_metrics.variance import computeVariancescore
from trainer_Cite import construct_model, Cora_test, get_num_misclassed

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
    err_list = []
    for exp_ID in range(20):
        # construct original model
        model, optimizer = construct_model(args, dataset)
        model = model.to(device)

        # load model
        savedpath_model = f"pretrained_all/pretrained_model/{args.type}_{args.data}/"
        model.load_state_dict(torch.load(os.path.join(savedpath_model, "model.pt"), map_location=device))

        # get selected index
        ratio = args.select_ratio
        retrain_length = len(test_index)
        select_num = int(ratio / 100 * retrain_length)
        select_index = Cora_select_functions(model, test_index, args.metrics, data, select_num, dataset)
        new_select_index = test_index[select_index]

        # get test accuracy of the small sample and the total test dataset
        sample_num_misclassed = get_num_misclassed(model, data, new_select_index)
        savedpath_num_test = f"pretrained_all/misclassed_num/{args.type}_{args.data}/num.npy"
        test_num_misclass = np.load(savedpath_num_test)
        detect_rate = sample_num_misclassed / test_num_misclass
        err_list.append(detect_rate)

        print(f'Exp: {exp_ID:03d},  Test mis is: {test_num_misclass:d}, sample mis is: {sample_num_misclassed:d}, detected ratio: {detect_rate:.4f}')
    err_avg = np.sum(err_list) / 20
    print("error rate is: ", err_avg)

    # save best accuracy
    savedpath_avg = f"misclassed_ratio/{args.type}_{args.data}/{args.metrics}/"
    if not os.path.isdir(savedpath_avg):
        os.makedirs(savedpath_avg, exist_ok=True)
    np.save(f"{savedpath_avg}/mis{ratio}.npy", err_avg)


def Cora_select_functions(model, test_index, metric, data, select_num, dataset):
    if metric == "deepgini":
        selected_index = DeepGini_metrics(model, test_index, data, select_num)
    elif metric == "random":
        selected_index = random_select(len(test_index), select_num)
    elif metric == "l_con":
        selected_index = least_confidence_metrics(model, test_index, data, select_num)
    elif metric == "entropy":
        selected_index = entropy_metrics(model, test_index, data, select_num)
    elif metric == "margin":
        selected_index = margin_metrics(model, test_index, data, select_num)
    elif metric == "variance":
        selected_index = computeVariancescore(model, test_index, data, select_num)
    elif metric == "mcp":
        selected_index = mcp_score(model, test_index, data, select_num, dataset.num_classes)

    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
