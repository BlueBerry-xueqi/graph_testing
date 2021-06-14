from scipy.stats.stats import mode
import torch
from torch_geometric.nn import global_add_pool
from torch.nn import Module
import collections

class SA_Model():
    def __init__(self, model, device) -> None:
        self.model = model
        self.activation = {}
        self.pool = global_add_pool
        self.device = device
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.clone().detach()
        return hook

    def register_layers(self, layer_names):
        for (name, module) in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook( self.get_activation(name) )
    
    def predict(self, dataset_loader):
        prediction = []
        ground_truth = []
        pre_labels = []
        for data in dataset_loader:
            data = data.to(self.device)
            outputs = self.model.compute(data)
            _, prediction_label = torch.max(outputs, dim=1)
            pre_labels.append( prediction_label )
            prediction.append(outputs)
            ground_truth.append(data.y)
        pre_prob = torch.cat( prediction, dim = 0).detach().cpu().numpy()
        pre_labels = torch.cat( pre_labels, dim = 0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truth, dim = 0).detach().cpu().numpy()
        return pre_prob, pre_labels, ground_truth


    def extract_intermediate_outputs(self, dataset_loader):
        pre = []
        ground_truth = []
        extracted_layers_outputs = collections.defaultdict(list)
        for data in dataset_loader:
            data = data.to(self.device)
            outputs = self.model.compute(data)
            _, prediction_label = torch.max(outputs, dim=1)
            pre.append(prediction_label)
            ground_truth.append(data.y)
            for layer_name in self.activation.keys():
                #print(self.activation[layer_name].shape)
                try:
                    imout = self.pool(self.activation[layer_name], data.batch)
                except:
                    imout = self.activation[layer_name]
                #print(imout.shape)
                extracted_layers_outputs[layer_name].append(imout)
        
        pre = torch.cat( pre, dim = 0)
        ground_truth = torch.cat(ground_truth, dim = 0)
        for k in extracted_layers_outputs:
            extracted_layers_outputs[k] = torch.cat(extracted_layers_outputs[k] , dim = 0).detach().cpu().numpy()
        return pre.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(), extracted_layers_outputs