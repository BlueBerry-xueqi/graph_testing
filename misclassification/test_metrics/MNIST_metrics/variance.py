import pdb

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def computeVariancescore(model, retrain_loader, select_num):
    scores = []
    with torch.no_grad():
        for data in retrain_loader:
            data = data.to(device)
            output = model(data)
            y_pred = torch.softmax(output, dim=1)
            result = torch.Tensor.cpu(y_pred).detach().numpy()

            var = np.var(result, axis=1)
            scores = np.concatenate((scores, var))

    select_index = np.argsort(scores)[:select_num]
    select_index = select_index.copy()
    return select_index
