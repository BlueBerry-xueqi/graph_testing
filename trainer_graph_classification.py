from models.pytorchtools import EarlyStopping,save_model_layer_bame
import torch
import os.path as osp
from models.gin_graph_classification import Net as GIN
from models.gat_graph_classification import Net as MuGIN
from models.gmt.nets import GraphMultisetTransformer
from models.gcn_graph_classification import Net as GCN
from models.pytorchtools import EarlyStopping,save_model_layer_bame
from torch_geometric.data import DataLoader
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup
from models.gmt.parser import Parser
from models.data import get_dataset
def construct_model(device, savedpath, model_name, load_pre=False):
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'TU')
    args = Parser().parse()
    dataset = get_dataset(args.data, normalize=args.normalize)
    if model_name == "gin":
        model = GIN(dataset.num_features, 64, dataset.num_classes, num_layers=3)
    if model_name == "gat":
        model = MuGIN(dataset.num_features,  32, dataset.num_classes)
    if model_name == "gmt":
        args.num_features, args.num_classes, args.avg_num_nodes = dataset.num_features, dataset.num_classes, np.ceil(np.mean([data.num_nodes for data in dataset]))
        model = GraphMultisetTransformer(args)
    if model_name == "gcn":
        model = GCN(dataset.num_features, 64, dataset.num_classes)
    if load_pre:
        print("Load Model Weights")
        model.load_state_dict( torch.load(f"{savedpath}/saved_model.pt", map_location=device) )
    return model,args


def train_and_save(model_name, savedpath="saved_model/gin_imdb_binary/"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, args = construct_model(device, savedpath, model_name, load_pre=False) 
    savedpath  = f"saved_model/{model_name}_{args.data}/"
    if not os.path.isdir( savedpath ):
        os.makedirs( savedpath, exist_ok=True )
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'TU')
    dataset = get_dataset(args.data, normalize=args.normalize) #TUDataset(path, name='COLLAB', transform=OneHotDegree(135)) #IMDB-BINARY binary classification
    dataset = dataset.shuffle()
    train_size = int(len(dataset)*0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split( dataset, [train_size, test_size] )
    os.makedirs(savedpath, exist_ok=True)
    torch.save(train_dataset, f"{savedpath}/train.pt")
    torch.save(test_dataset, f"{savedpath}/test.pt")
    test_loader = DataLoader(test_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)
  
    save_model_layer_bame(model, f"{savedpath}/layers.json")
    model = model.to(device)
    #model = torch.jit.script(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # initialize the early_stopping object
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=savedpath)
    if args.lr_schedule:
        scheduler = get_cosine_schedule_with_warmup(optimizer,  args.patience * len(train_loader), args.num_epochs * len(train_loader))
    def train():
        model.train()
        total_loss = 0.
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
            if args.lr_schedule:
                scheduler.step()
        return total_loss / len(train_dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_correct = 0
        val_loss = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            val_loss += loss.item() * data.num_graphs
            pred = out.max(dim=1)[1]
            total_correct += pred.eq(data.y).sum().item()

        return total_correct / len(loader.dataset), val_loss/len(loader.dataset)

    

    for epoch in range(1, args.num_epochs ):
        loss = train()
        train_acc, train_loss = test(train_loader)
        test_acc, test_loss = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
            f'Train: {train_acc:.4f}, Test: {test_acc:.4f}')
        early_stopping(test_loss, model, performance={"train_acc":train_acc, "test_acc":test_acc, "train_loss":train_loss, "test_loss":test_loss})
        if early_stopping.early_stop:
            print("Early stopping")
            break

import argparse
if __name__ == "__main__":
    modelparser = argparse.ArgumentParser()
    modelparser.add_argument("-t", "--type", type=str, default="mugin", help="gin, mugin")
    #parser.add_argument("-s", "--savedpath", type=str, default="saved_model/mugin_imdb_binary/")
    modelargs = modelparser.parse_known_args()[0]
    
    train_and_save(modelargs.type )
