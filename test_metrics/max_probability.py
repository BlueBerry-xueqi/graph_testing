import pdb
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def max_score(model, target_loader, select_num):
    scores = []
    with torch.no_grad():
        for data in target_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            y_pred = torch.softmax(output, dim=1)
            score = np.max(np.exp(torch.Tensor.cpu(y_pred).detach().numpy()), axis=1)
            scores = np.concatenate((scores, score))

    scores_sort = np.argsort(scores)[::-1]
    select_index = scores_sort[-select_num:]
    select_index = torch.from_numpy(np.flip(select_index, axis=0).copy())
    return select_index
