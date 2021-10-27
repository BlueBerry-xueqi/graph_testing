
from models.gin_graph_classification import Net as GIN
from models.gat_graph_classification import Net as MuGIN
from models.gmt.nets import GraphMultisetTransformer
from models.gcn_graph_classification import Net as GCN
from models.gnn_cora import Net as COG
from models.pytorchtools import EarlyStopping, save_model_layer_bame
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
from parser import Parser
from models.data import get_dataset



# 用来训练模型的方法
def construct_model(args, dataset, device, savedpath, model_name, load_pre=False):
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'TU')
    # args = Parser().parse()
    # # 得到数据集
    # dataset = get_dataset(args.data, normalize=args.normalize)
    if model_name == "gin":
        model = GIN(dataset.num_features, 64, dataset.num_classes, num_layers=3)
    elif model_name == "gat":
        model = MuGIN(dataset.num_features, 32, dataset.num_classes)
    elif model_name == "gmt":
        args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(
            np.mean([data.num_nodes for data in dataset]))
        model = GraphMultisetTransformer(args)
    elif model_name == "gcn":
        model = GCN(dataset.num_features, 64, dataset.num_classes)
    if args.data == "Cora":
        model = COG(dataset)
    if load_pre:
        print("Load Model Weights")
        model.load_state_dict(torch.load(f"{savedpath}/pretrained_model.pt", map_location=device))
    return model, args


def train_and_save(args):
    # 放在cpu上跑
    model_name = args.type
    savedpath = args.savedpath
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("args.data是：" + args.data)
    dataset = get_dataset(args.data,
                          normalize=args.normalize)
    model = construct_model(args, dataset, device, savedpath, model_name, load_pre=False)
    if args.data == "Cora":
        model = model[0]
    # 设置一个saved path，放置的是pretrained 模型
    savedpath = f"pretrained_model/{model_name}_{args.data}/"
    # 假如说不存在这个savedpath，那么创建这个savedpath
    if not os.path.isdir(savedpath):
        os.makedirs(savedpath, exist_ok=True)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'TU')
    # TUDataset(path, name='COLLAB', transform=OneHotDegree(135)) #IMDB-BINARY binary classification
    # 将dataset 重新洗一下
    dataset = dataset.shuffle()
    # 得到训练集的长短
    train_size = int(len(dataset) * 0.8)
    # 得到测试选择集的长短
    testselection_size = int(len(dataset) * 0.1)
    # 得到测试集的长短
    test_size = len(dataset) - train_size - testselection_size

    # check whether the specified path is an existing regular file or not.
    # 检查一个路径是不是一个正常的文件，假如说不正常的话，直接将应该有的值存储
    if not (os.path.isfile(f"{savedpath}/train_index.pt") and os.path.isfile(
            f"{savedpath}/test_selection_index.pt") and os.path.isfile(f"{savedpath}/test_index.pt")):
        # 得到一个索引，这个索引是0到数据集长度的一个随机数
        index = torch.randperm(len(dataset))
        # 得到训练集合，测试数据集和测试集
        train_dataset_index, test_selection_dataset_index, test_dataset_index = index[:train_size], index[
                                                                                                    train_size:train_size + testselection_size], index[
                                                                                                                                                 train_size + testselection_size:]
        # 根据savedpath来创建一个文件夹
        os.makedirs(savedpath, exist_ok=True)
        # 将训练集的索引存储在这个文件夹下面
        torch.save(train_dataset_index, f"{savedpath}/train_index.pt")
        # 将测试数据的索引存储在这个文件夹下面
        torch.save(test_selection_dataset_index, f"{savedpath}/test_selection_index.pt")
        # 将测试集的索引放在测试文件夹下面
        torch.save(test_dataset_index, f"{savedpath}/test_index.pt")


    else:
        # 加载训练集，测试数据选择集和测试集
        train_dataset_index = torch.load(f"{savedpath}/train_index.pt")
        test_selection_index = torch.load(f"{savedpath}/test_selection_index.pt")
        test_dataset_index = torch.load(f"{savedpath}/test_index.pt")
        # 通过索引得到训练集，测试集和测试数据选择集的真正的数据
        train_dataset, test_selection_dataset, test_dataset = dataset[train_dataset_index], dataset[
            test_selection_index], dataset[test_dataset_index]
        # 通过dataset得到测试的loader
        print(type(train_dataset))
        if args.data != "Cora":
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=128)
            # 假如说趋于平稳的话，那么久提前结束
            early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=savedpath)
        # 将训练好的模型放到layers.json下面
        # save_model_layer_bame(model, f"{savedpath}/layers.json")
        # 将模型放到cpu里面
        model = model.to(device)
        # model = torch.jit.script(model)

        # 将模型进行优化
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # initialize the early_stopping object

    #     假如说
    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.patience * len(train_loader),
                                                    args.num_epochs * len(train_loader))

    # 用于训练模型的方法
    def train():
        # 将模型的模式调整为训练模式
        model.train()
        # 最开始的时候的loss是0
        total_loss = 0.
        # 对于train_loader中的数据
        for data in train_loader:
            # 首先把数据放在device(cpu)中
            data = data.to(device)
            # 将数据放入优化器中
            optimizer.zero_grad()
            # 根据训练的数据的x和edge的值得到模型out
            # Perform a single forward pass. 一个简单的向前训练的过程
            # GCN(dataset.num_features, 64, dataset.num_classes)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
            if args.lr_schedule:
                scheduler.step()
        return total_loss / len(train_dataset)

    def trainCora():
        # 将模型的模式调整为训练模式
        dataT = dataset[0]
        model.train()
        # 最开始的时候的loss是0
        optimizer.zero_grad()
        total_loss = F.nll_loss(model()[dataT.train_mask], dataT.y[dataT.train_mask])
        total_loss.backward()
        optimizer.step()
        return total_loss

    # test的函数
    @torch.no_grad()
    def testCora(dataset):
        model.eval()
        dataT = dataset[0]
        log_probs, test_acc = model(), []
        for _, mask in dataT('test_mask'):
            pred = log_probs[mask].max(1)[1]
            test_acc = pred.eq(dataT.y[mask]).sum().item() / mask.sum().item()
        return test_acc

    @torch.no_grad()
    def test(loader):
        # 将模型调整到测试状态
        model.eval()
        # 总的正确的值是0
        total_correct = 0
        # loss是0
        val_loss = 0
        # 对于在test loader中的数据
        for data in loader:
            # 将数据放在device中
            data = data.to(device)
            # forward的过程
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            val_loss += loss.item() * data.num_graphs
            pred = out.max(dim=1)[1]
            total_correct += pred.eq(data.y).sum().item()

        return total_correct / len(loader.dataset), val_loss / len(loader.dataset)

    # 每一个训练过程执行5次
    for epoch in range(1, args.num_epochs):
        print("epoch: {epoch}")
        # 得到训练的loss
        if args.data != "Cora":
            loss = train()
            train_acc, train_loss = test(train_loader)
            # test准确率和test的loss
            test_acc, test_loss = test(test_loader)
        else:
            loss = trainCora()
            # test准确率和test的loss
            test_acc = testCora(dataset)
        # 得到

        # 打印出来准确率
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
        #       f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        if args.data != "Cora":
            early_stopping(test_loss, model,
                       performance={"train_acc": train_acc, "test_acc": test_acc, "train_loss": train_loss,
                                    "test_loss": test_loss})
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if args.data == "Cora":
        torch.save(model.state_dict(), os.path.join(savedpath, "pretrained_model.pt"))



if __name__ == "__main__":
    # modelparser = argparse.ArgumentParser(description="Training parser")
    #
    # modelparser.add_argument('--type', default='gin', type=str,
    #                          choices=['gin', 'gat', 'gmt', 'gcn'], required=False)
    # modelparser.add_argument('--data', type=str, default='DD',
    #                          choices=['DD', 'Cora', 'PTC_MR'],
    #                          help='dataset type')
    args = Parser().parse()
    # parser.add_argument("-s", "--savedpath", type=str, default="pretrained_model/mugin_imdb_binary/")
    # modelargs = modelparser.parse_known_args()[0]
    train_and_save(args)
