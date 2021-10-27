import argparse

from models.gin_graph_classification import Net as GIN
from models.gat_graph_classification import Net as MuGIN
from models.gmt.nets import GraphMultisetTransformer
from models.gcn_graph_classification import Net as GCN
from models.pytorchtools import EarlyStopping
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
from parser import Parser
from models.data import get_dataset
from test_metrics.CES import selectsample
from test_metrics.DeepGini import deepgini_score
from test_metrics.MCP import MCP_score
from test_metrics.max_probability import max_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 输入要被选择的数据
def select_functions(model, target_loader, target_data, select_num, metric, savedpath, ncl=None):
    # 调用不同的metrics
    if metric == "CES":
        scores, _, _ = selectsample(args, model, target_loader)
    if metric == "DeepGini":
        scores = deepgini_score(model, target_loader)
    if metric == "max_probability":
        scores, _, _ = max_score(model, target_loader)
    if metric == "MCP":
        scores, _, _ = MCP_score(model, target_loader, ncl)
    if metric == "ModelWrapper":
        scores, _, _ = deepgini_score(model, target_loader)
    if metric == "nc":
        scores, _, _ = deepgini_score(model, target_loader)
    if metric == "sa":
        scores, _, _ = deepgini_score(model, target_loader)
    if metric == "sihoutete":
        scores, _, _ = deepgini_score(model, target_loader)

    # 根据score进行排序，然后得到score最大的前num条data的索引，把scores按照从大到小的顺序进行排列，然后找到前selected_num位
    selected_index = np.argsort(scores)[:select_num]
    # 将选择的数据的index保存，将选择的测试数据的index放到路径里面
    torch.save(selected_index, f"{savedpath}/selectedFunctionIndex.pt")
    # 通过索引得到被选择的数据   # DataLoader是PyTorch中的一种数据类型

    # 从Dataloader里面得到selected_loader
    selected_loader = DataLoader(target_data[selected_index], batch_size=128)

    return selected_loader


# 用来load data的函数
def loadData(args, savedpath=None):
    # 得到数据集
    dataset = get_dataset(args.data,
                          normalize=args.normalize)  # TUDataset(path, name='COLLAB', transform=OneHotDegree(135)) #IMDB-BINARY binary classification
    # shuffle数据集
    dataset = dataset.shuffle()
    # 训练数据的size是数据集size的0.8
    train_size = int(len(dataset) * 0.8)
    # 选择的当作test data的数据的size是总数据集的0.1
    testselection_size = int(len(dataset) * 0.1)
    # 测试数据集的size是剩下的数据集
    test_size = len(dataset) - train_size - testselection_size

    # 假如说没有生成文件的前提下，生成文件
    if not (os.path.isfile(f"{savedpath}/train.pt") and os.path.isfile(f"{savedpath}/test_selection_index.pt") and os.path.isfile(f"{savedpath}/test_index.pt")):
        # Returns a random permutation of integers from 0 len(dataset) -1
        index = torch.randperm(len(dataset))

        train_dataset_index, test_selection_dataset_index, test_dataset_index = index[:train_size], index[
                                                                                                    train_size:train_size + testselection_size], index[
                                                                                                                                                 train_size + testselection_size:]
        # os.makedirs() 方法用于递归创建目录
        # 语法：os.makedirs(path, mode=0o777)
        # path - - 需要递归创建的目录，可以是相对或者绝对路径。。
        # mode - - 权限模式

        os.makedirs(savedpath, exist_ok=True)
        # 保存所有的数据集
        torch.save(train_dataset_index, f"{savedpath}/train_index.pt")
        torch.save(test_selection_dataset_index, f"{savedpath}/test_selection_index.pt")
        torch.save(test_dataset_index, f"{savedpath}/test_index.pt")
        print("here good ")
    else:
        print("执行到这里了")
        train_dataset_index = torch.load(f"{savedpath}/train_index.pt")
        test_selection_index = torch.load(f"{savedpath}/test_selection_index.pt")
        test_dataset_index = torch.load(f"{savedpath}/test_index.pt")

    train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[test_selection_index], \
                                                          dataset[test_dataset_index]

    # 得到训练集的size
    train_size = len(train_dataset_index)
    # test loader是测试数据集
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    test_selection_loader = DataLoader(test_selection_dataset, batch_size=100, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    return train_loader, test_selection_loader, test_loader, test_selection_dataset, train_size


# 用来构建模型的函数，

def construct_model(savedpath, model_name, load_pre=False):
    # 得到参数
    args = Parser().parse()
    # 得到数据集
    dataset = get_dataset(args.data, normalize=args.normalize)
    # 判断属于哪一个模型，让model等于那个模型
    if model_name == "gin":
        model = GIN(dataset.num_features, 64, dataset.num_classes, num_layers=3)
    if model_name == "gat":
        model = MuGIN(dataset.num_features, 32, dataset.num_classes)
    if model_name == "gmt":
        args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
            np.mean([data.num_nodes for data in dataset]))
        model = GraphMultisetTransformer(args)
    if model_name == "gcn":
        model = GCN(dataset.num_features, 64, dataset.num_classes)
    if load_pre:
        print("Load Model Weights")
        model.load_state_dict(torch.load(f"{savedpath}/pretrained_model.pt", map_location=device))
    #     返回模型和参数
    return model, args


# 重新训练模型并保存
# 主要思路：
# 1. 首先拿到之前train好的模型
# 2. 将测试挑选数据拿出来，重新训练之前的模型
def retrain_and_save(args, model, selected_loader, train_loader, test_loader, train_size, selected_num, savedpath):
    # # 将model保存在layer.json层
    # save_model_layer_bame(model, f"{savedpath}/layers.json")
    # 加载优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=savedpath)
    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * len(train_loader),
                                                    args.num_epochs * len(train_loader))

    # 重新训练模型，参数是训练的数据和被选择的数据，训练的方法是：一般重新训练5次，每一次训练出来一个模型，然后在这个模型的基础上再用selection_data训练一遍
    # 这样一共训练5-10次，得出来最终的模型
    def retrain(train_loader, selected_loader, train_size, selected_num):
        # train the model
        model.train()
        # total loss is defined 0
        total_loss = 0.0
        retrainLoader = train_loader + selected_loader
        retrainLoader = retrainLoader.shuffle()

        # retrain model
        for data in retrainLoader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
            if args.lr_schedule:
                scheduler.step()

        return total_loss / (train_size + selected_num)

    # 这里的loader是test_
    @torch.no_grad()
    def test(loader):
        # it is supposed to allow me to "evaluate my model"
        # turn off some layers during the model evaluation process
        model.eval()

        # 定义一个初始的正确率
        total_correct = 0
        # 定义一个初始的loss
        val_loss = 0
        # 对于在test data里面的数据，首先将数据放在gpu中
        for data in loader:
            data = data.to(device)
            # 根据数据，训练模型
            out = model(data.x, data.edge_index, data.batch)
            # 得到训练模型的损失
            loss = F.nll_loss(out, data.y)
            # 累加损失
            val_loss += loss.item() * data.num_graphs
            pred = out.max(dim=1)[1]
            # total_correct 累加
            total_correct += pred.eq(data.y).sum().item()

        return total_correct / len(loader.dataset), val_loss / len(loader.dataset)

    for epoch in range(args.retrain_epochs):
        # 重新训练模型，得到模型训练损失
        loss = retrain(train_loader, selected_loader, train_size, selected_num)
        print(loss)
        # 得到测试的准确性和损失
        test_acc, test_loss = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ')
        # 假如趋于稳定那么提早进行stop
        early_stopping(test_loss, model, performance={"test_acc": test_acc, "test_loss": test_loss})
        if early_stopping.early_stop:
            print("Early stopping")
            break


def main(model_name, dataName, metric, exp, modelRoot):
    print("main success")
    # load pre-trained model
    # 得到模型的路径
    modelPath = modelRoot + "gin_{0}".format(dataName)
    # 得到模型和模型参数
    model, args = construct_model(modelPath, model_name, load_pre=True)
    # 放在gpu里面跑模型
    model = model.to(device)
    model.eval()
    # path to save retrained model 将模型存储
    savedpath = f"retrained_model/{model_name}_{dataName}_{metric}/exp-{exp}/"
    # retrained_model/gin_DD/exp-0/{metric}.pt
    if not os.path.isdir(savedpath):
        os.makedirs(savedpath, exist_ok=True)

    # load data and select a subset of data for retraining
    # 加载训练loader，selection loader， 测试loader，测试selection loader，训练size
    train_loader, test_selection_loader, test_loader, test_selection_data, train_size = loadData(args, modelPath)
    # 通过metrics的selection function得到selection loader
    selected_loader = select_functions(model, test_selection_loader, test_selection_data, args.selected_num,
                                       args.metrics, savedpath)

    # 调用retrain方法，重新训练模型
    # retrain model using selected data and save it

    total_correct, loss = retrain_and_save(args, model, selected_loader, train_loader, test_loader, train_size,
                                           args.select_num,savedpath)
    # 存储正确性和loss
    torch.save(total_correct, f"{savedpath}/accuracy.pt")
    torch.save(loss, f"{savedpath}/loss.pt")


import argparse

if __name__ == "__main__":
    args = Parser().parse()

    main(args.type, args.data, args.metrics, args.exp, "pretrained_model/")
