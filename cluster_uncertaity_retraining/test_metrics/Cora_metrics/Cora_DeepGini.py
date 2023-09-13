import pdb

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def DeepGini_metrics(model, retrain_index, data, select_num):
    scores = []
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        gini_score = 1 - np.sum(np.power(torch.Tensor.cpu(y_pred).detach().numpy(), 2), axis=1)
        scores = np.concatenate((scores, gini_score))

    # sort through ascending order
    select_index = np.argsort(scores)[::-1][:select_num]
    select_index = select_index.copy()
    return select_index
