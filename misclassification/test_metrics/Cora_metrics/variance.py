import pdb

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def computeVariancescore(model, retrain_index,data, select_num):
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        x = torch.Tensor.cpu(y_pred).detach().numpy()
        var = np.var(x, axis=1)

    select_index = np.argsort(var)[:select_num]
    select_index = select_index.copy()
    return select_index
