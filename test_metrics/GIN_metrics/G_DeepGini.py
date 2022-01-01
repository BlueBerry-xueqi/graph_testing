import pdb

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def DeepGini_metrics(model, retrain_loader, select_num):
    # pre_prob, pre_labels, ground_truth = model.predict(target_data)
    scores = []
    with torch.no_grad():
        for data in retrain_loader:
            data = data.to(device)
            output = model(data)
            y_pred = torch.softmax(output, dim=1)
            gini_score = 1 - np.sum(np.power(torch.Tensor.cpu(y_pred).detach().numpy(), 2), axis=1)
            scores = np.concatenate((scores, gini_score))

    # sort through ascending order
    select_index = np.argsort(scores)[::-1][:select_num]
    select_index = select_index.copy()
    return select_index