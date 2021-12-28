import copy
import pdb
import sys

import numpy
from torch_geometric.loader import DataLoader, DenseDataLoader

from models.geometric_models.K_center_greedy import k_center_greedy_selection
from test_metrics.Entropy import entropy
from test_metrics.MCP import MCP_score
from test_metrics.Margin import margin_score
from test_metrics.least_confidence import least_confidence_score
from test_metrics.new_metrics.BALD import BALD_selection
from test_metrics.new_metrics.Kmeans_selection import Kmeans_selection
from trainer_graph_classification import construct_model, Cora_test, Cora_train
from models.geometric_models.mnist_nn_conv import Net as NN, train as MNIST_train, test as MNIST_test
from models.geometric_models.mutag_gin import Net as GIN_TU, train as TU_train, test as TU_test
from models.geometric_models.GraphNN import Net as GraphNN, train as GrapNN_train, test as GraphNN_test
# from models.geometric_models.proteins_gmt import Net as GMT, train as GMT_train, test as GMT_test
from models.geometric_models.proteins_diff_pool import Net as SAGE, train as SAGE_train, test as SAGE_test
from models.geometric_models.mem_pool import Net as mem_pool, train as mem_train, test as mem_test
from test_metrics.Cora_metrics.Cora_Kmeans_selection import Kmeans_selection as Cora_Kmeans_selection
from test_metrics.Cora_metrics.Cora_BALD import BALD_selection as Cora_BALD_selection

import os
import torch
import numpy as np
from parser import Parser
from test_metrics.DeepGini import deepgini_score
from test_metrics.random import random_select

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

        if args.type == "SAGE":
            test_loader = DenseDataLoader(test_dataset, batch_size=20)
            val_loader = DenseDataLoader(val_dataset, batch_size=20)
            train_loader = DenseDataLoader(train_dataset, batch_size=20)
            retrain_loader = DenseDataLoader(train_dataset, batch_size=20)
        else:
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            retrain_loader = DataLoader(retrain_dataset, batch_size=128, shuffle=False)

    return dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset


def retrain_and_save(args):
    if args.data == "Cora":
        data, dataset, train_index, val_index, test_index, retrain_index = Cora_loadData(args)
    else:
        dataset, train_index, test_index, retrain_index, train_loader, val_loader, test_loader, retrain_loader, train_dataset, val_dataset, test_dataset, retrain_dataset = loadData()
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
        model, optimizer = construct_model(args, dataset)
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
        savedpath_index = f"retrained_all/selected_index/{args.metrics}/{args.type}_{args.data}/all_index.pt"
        if not os.path.isfile(savedpath_index):
            os.makedirs(savedpath_index, exist_ok=True)
            # get retrain loader for select functions
            select_num = int(ratio / 100 * len_retrain_set)
            if args.data == "Cora":
                total_num = len_retrain_set
                cora_data =data
                all_index = Cora_select_functions(model, retrain_index, total_num, args.metrics, cora_data, select_num)
            else:
                all_index = select_functions(model, retrain_index, len_retrain_set,
                                             args.metrics, None, False, retrain_loader, retrain_dataset, select_num)
            torch.save(all_index, os.path.join(savedpath_index, "all_index.pt"))
        else:
            # if the indexes are already saved, use the index directly.
            all_index = torch.load(f"{savedpath_index}all_index.pt")
            print("sorted indexed loaded")

        # take this length from the retrain set
        length_current = int(ratio / 100 * len_retrain_set)
        # length_pre = int(ratio_pre / 100 * len_retrain_set)

        select_index = all_index[:length_current]
        select_index = numpy.array(select_index)
        select_index = torch.from_numpy(select_index)
        new_train_index = torch.cat((select_index, train_index), -1)

        # initiallize the best accuracy
        best_acc = 0
        best_val_acc = test_acc = 0
        for epoch in range(args.retrain_epochs):
            if args.data == "Cora":
                Cora_train(model, optimizer, data, new_train_index)
                train_acc, val_acc, test_acc = Cora_test(model, data, new_train_index, val_index, test_index)
            if args.data == "MNIST":
                new_train_dataset = dataset[new_train_index]
                new_train_loader = DataLoader(new_train_dataset, batch_size=128, shuffle=True)

                MNIST_train(epoch, new_train_loader, model, optimizer)
                train_acc = MNIST_test(new_train_loader, new_train_dataset, model)
                val_acc = MNIST_test(val_loader, val_dataset, model)
                test_acc = MNIST_test(test_loader, test_dataset, model)
            if args.type == "GraphNN":
                new_train_dataset = dataset[new_train_index]
                new_train_loader = DataLoader(new_train_dataset, batch_size=128, shuffle=True)
                GrapNN_train(model, optimizer, new_train_loader, new_train_dataset)
                train_acc = GraphNN_test(new_train_loader, model)
                val_acc = GraphNN_test(val_loader, model)
                test_acc = GraphNN_test(test_loader, model)
            if args.type == "GIN":
                new_train_dataset = dataset[new_train_index]
                new_train_loader = DataLoader(new_train_dataset, batch_size=128, shuffle=True)

                TU_train(model, new_train_loader, optimizer)
                train_acc = TU_test(new_train_loader, model)
                val_acc = TU_test(val_loader, model)
                test_acc = TU_test(test_loader, model)
            # if args.type == "GMT":
            #     new_train_dataset = dataset[new_train_index]
            #     new_train_loader = DataLoader(new_train_dataset, batch_size=128, shuffle=True)
            #     GMT_train(model, new_train_loader, new_train_dataset, optimizer)
            #     train_acc = GMT_test(new_train_loader, model)
            #     val_acc = GMT_test(val_loader, model)
            #     tmp_test_acc = GMT_test(test_loader, model)
            if args.type == "SAGE":
                new_train_dataset = dataset[new_train_index]
                new_train_loader = DenseDataLoader(new_train_dataset, batch_size=20)

                SAGE_train(model, optimizer, new_train_loader, new_train_dataset)
                train_acc = SAGE_test(new_train_loader, model)
                val_acc = SAGE_test(val_loader, model)
                test_acc = SAGE_test(test_loader, model)
            if args.type == "mempool":
                new_train_dataset = dataset[new_train_index]
                new_train_loader = DataLoader(new_train_dataset, batch_size=20)
                mem_train(new_train_loader, model, optimizer)
                train_acc = mem_test(new_train_loader, model)
                val_acc = mem_test(val_loader, model)
                test_acc = mem_test(test_loader, model)

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     test_acc = tmp_test_acc

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


def select_functions(model, retrain_index, total_num, metric, cora_data, is_Cora, retrain_loader, retrain_dataset,
                     select_num):
    if args.type == "mempool":
        is_mem = True
    else:
        is_mem = False

    if metric == "DeepGini":
        selected_index = deepgini_score(model, retrain_index, total_num, cora_data, is_Cora, retrain_loader, is_mem)
    elif metric == "random":
        selected_index = random_select(len(retrain_index), total_num)
    elif metric == "l_con":
        selected_index = least_confidence_score(model, retrain_index, total_num, cora_data, is_Cora, retrain_loader,
                                                is_mem)
    elif metric == "entropy":
        selected_index = entropy(model, retrain_index, total_num, cora_data, is_Cora, retrain_loader, is_mem)
    elif metric == "margin":
        selected_index = margin_score(model, retrain_index, total_num, cora_data, is_Cora, retrain_loader, is_mem)

    # new added metrics
    elif metric == "MCP":
        selected_index = MCP_score(model, retrain_loader, total_num, is_mem, cora_data, is_Cora, retrain_index,
                                   ncl=retrain_dataset.num_classes)
    elif metric == "BALD":
        selected_index = BALD_selection(model, retrain_loader, select_num, total_num, is_mem, cora_data, is_Cora,
                                        retrain_index)

    elif metric == "k-center":
        lb = np.zeros(len(retrain_dataset), dtype=bool)
        selected_index = k_center_greedy_selection(model, retrain_loader, select_num, total_num, lb, cora_data, is_Cora,
                                                   retrain_index, is_mem)

    elif metric == "K-Means":
        selected_index = Kmeans_selection(model, retrain_loader, total_num, select_num, retrain_index, is_Cora,
                                          cora_data, is_mem)

    return selected_index


def Cora_select_functions(model, retrain_index, total_num, metric, cora_data, select_num):

    # if metric == "k-center":
    #     lb = np.zeros(len(retrain_index), dtype=bool)
    #     selected_index = k_center_greedy_selection(model, select_num, total_num, lb, cora_data, retrain_index)

    if metric == "BALD":
        selected_index = Cora_BALD_selection(model, select_num, total_num,  cora_data, retrain_index)

    elif metric == "random":
        selected_index = random_select(len(retrain_index), total_num)

    elif metric == "K-Means":
        selected_index = Cora_Kmeans_selection(model, select_num, retrain_index, cora_data)

    return selected_index


if __name__ == "__main__":
    args = Parser().parse()
    retrain_and_save(args)
