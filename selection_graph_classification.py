import copy
import pdb
import sys

from test_metrics.CES import conditional_sample
from test_metrics.Entropy import entropy
from test_metrics.Margin import margin_score
from test_metrics.max_probability import max_score
from test_metrics.sa import fetch_dsa, fetch_lsa
from trainer_graph_classification import construct_model, test
from torch_geometric.data import DataLoader
import os
import torch
import numpy as np
from parser import Parser
from test_metrics.DeepGini import deepgini_score
from test_metrics.random import random_select
from trainer_graph_classification import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loadData(args):
    print("Dataset loading...")
    savedpath = f"pretrained_all/pretrained_data/{ args.type}_{args.data}"
    if os.path.isdir(savedpath):
        dataset = torch.load(f"{savedpath}/dataset.pt")

        n = len(dataset) // 10
        train_dataset = dataset[:n * 4]
        retrain_dataset = dataset[n * 4:n * 8]
        test_dataset = dataset[n * 8:]

        train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)
        retrain_loader = DataLoader(retrain_dataset, batch_size=60, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)
        print("Dataset loaded successful!")

    else:
        print("File pretrained does not exist!!!")
    return dataset, train_dataset, test_dataset, retrain_dataset, train_loader, retrain_loader, test_loader


def retrain_and_save(args):
    dataset, train_dataset, test_dataset, retrain_dataset, train_loader, retrain_loader, test_loader = loadData(args)
    len_retrain_set = (len(dataset) / 10 * 4)
    select_num = int(len_retrain_set * (args.select_ratio / 100))

    for exp_ID in range(args.exp):
        print("exp", exp_ID, "...")
        savedpath_retrain = f"retrained_all/retrained_model/{args.metrics}/{args.type}_{args.data}/{exp_ID}/"
        if not os.path.isdir(savedpath_retrain):
            os.makedirs(savedpath_retrain, exist_ok=True)

        # get original model
        model, early_stopping, optimizer = construct_model(args, dataset, train_loader, savedpath_retrain)
        model = model.to(device)

        # create savedpath pretrain
        savedpath_pretrain = f"pretrained_all/pretrained_model/{args.type}_{args.data}/model.pt"

        # load model from savedpath pretrain
        if os.path.isfile(savedpath_pretrain):
            model.load_state_dict(torch.load(savedpath_pretrain, map_location=device))
        else:
            print("pre-trained model not exists")
            sys.exit()

        # get selected dataset

        selected_index = select_functions(model, retrain_loader, retrain_dataset, select_num, args.metrics,
                                          train_loader)
        # combine training set and selected set
        select_dataset = retrain_dataset[selected_index]
        new_dataset = torch.utils.data.ConcatDataset([train_dataset, select_dataset])
        train_loader_new = DataLoader(dataset=new_dataset, batch_size=60, shuffle=True)

        # retrain process
        best_acc = 0
        for epoch in range(args.retrain_epochs):
            loss = train(model, train_loader_new, optimizer, train_dataset)
            # if args.lr_schedule:
            #     scheduler.step()
            # test model
            train_acc, train_loss = test(train_loader_new, model)
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

            print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, 'f'Test Acc: {test_acc:.5f}')

        torch.save(best_model, os.path.join(savedpath_retrain, "model.pt"))
        savedpath_acc = f"retrained_all/train_accuracy/{args.metrics}/{args.type}_{args.data}/{exp_ID}"
        if not os.path.isdir(savedpath_acc):
            os.makedirs(savedpath_acc, exist_ok=True)
        print(best_acc, "======")
        np.save(f"{savedpath_acc}/test_accuracy_ratio{args.select_ratio}.npy", best_acc)


def select_functions(model, target_loader, target_data, select_num, metric, train_loader):
    if metric == "DeepGini":
        selected_index = deepgini_score(model, target_loader, select_num)
    elif metric == "random":
        selected_index = random_select(len(target_data), select_num)
    elif metric == "max_probability":
        selected_index = max_score(model, target_loader, select_num)
    elif metric == "Entropy":
        selected_index = entropy(model, target_loader, select_num)
    elif metric == "margin":
        selected_index = margin_score(model, target_loader, select_num)
    elif metric == "dsa":
        selected_index = fetch_dsa(model, train_loader, target_loader, "test_selection_loader", [], select_num)
    elif metric == "lsa":
        selected_index = fetch_lsa(model, target_loader, select_num)

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
