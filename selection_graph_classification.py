import sys
from trainer_graph_classification import construct_model, loadData, test, testCora
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from parser import Parser
from test_metrics.CES import CES_selection, selectsample
from test_metrics.DeepGini import deepgini_score
from test_metrics.random import random_select

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def retrain_and_save(args):
    # load data
    dataset, train_loader, test_loader, test_selection_loader, train_dataset_index, test_selection_dataset, test_selection_index = loadData(args)
    # get model name and type
    model_name = args.type

    for exp_ID in range(args.exp):
        savedpath_retrain = f"retrained_all/retrained_model/{args.metrics}/{model_name}_{args.data}/{exp_ID}/"
        if not os.path.isdir(savedpath_retrain):
            os.makedirs(savedpath_retrain, exist_ok=True)
        # get original model
        model, _, optimizer, scheduler = construct_model(args, dataset, train_loader, savedpath_retrain)
        model = model.to(device)
        savedpath_pretrain = f"pretrained_all/pretrained_model/{model_name}_{args.data}/model.pt"
        if os.path.isfile(savedpath_pretrain):
            model.load_state_dict(torch.load(savedpath_pretrain, map_location=device))
        else:
            print("pre-trained model not exists")
            sys.exit()

        # get selected dataset
        select_num = int(np.ceil(len(test_selection_index) * args.select_ratio / 100))
        selected_index = select_functions(model, test_selection_loader, test_selection_dataset, select_num, args.metrics, ncl=None)

        # combine training set and selected set
        train_index_new = np.concatenate((train_dataset_index, test_selection_index[selected_index]))
        train_loader_new = DataLoader(dataset[train_index_new], batch_size=128, shuffle=True)
            
        if args.data != "Cora":
            best_acc, _ = test(test_loader, model)
        # retrain process
        for epoch in range(args.epochs):
            if args.data != "Cora":
                # train model
                model.train()
                total_loss = 0.0
                for data in train_loader_new:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.batch)
                    loss = F.nll_loss(out, data.y)
                    loss.backward()
                    total_loss += loss.item() * data.num_graphs
                    optimizer.step()
                    if args.lr_schedule:
                        scheduler.step()
                # test model
                acc, _ = test(test_loader, model)
            else:
                print("Not implemented")
                sys.exit()
#                 dataT = dataset[0]
#                 model.train()
#                 optimizer.zero_grad()
#                 total_loss = F.nll_loss(model()[dataT.train_mask], dataT.y[dataT.train_mask])
#                 total_loss.backward()
#                 optimizer.step()
#                 acc = testCora(dataset, model)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(savedpath_retrain, "model.pt"))

            savedpath_acc = f"retrained_all/train_accuracy/{model_name}_{args.data}/{exp_ID}"
            if not os.path.isdir(savedpath_acc):
                os.makedirs(savedpath_acc, exist_ok=True)
            np.save(f"{savedpath_acc}/test_accuracy.npy", best_acc)


# select retrain data based on metrics
def select_functions(model, target_loader, target_data, select_num, metric, ncl=None):
    if metric == "DeepGini":
        selected_index = deepgini_score(model, target_loader, select_num)
    elif metric == "random":
        selected_index = random_select(len(target_data), select_num)
    # elif metric == "CES":
    #     selected_index = CES_selection(model, target_data, select_num)
    # if metric == "max_probability":
    #     scores, _, _ = max_score(model, target_loader)
    # if metric == "MCP":
    #     scores = MCP_score(model, target_loader, ncl)
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
