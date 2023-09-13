import pdb
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def least_confidence_metrics(model, retrain_index, data, select_num):
    scores = []
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        predic_prob = torch.Tensor.cpu(y_pred).detach().numpy()
        max_pre = predic_prob.max(1)
        scores = np.concatenate((scores, max_pre))
    select_index = np.argsort(scores)[::-1][:select_num]
    select_index = select_index.copy()
    return select_index
