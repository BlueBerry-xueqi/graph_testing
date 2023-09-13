import pdb

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predict_list(retrain_loader, model):
    predict_list = np.array([[0, 0, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        for data in retrain_loader:
            data = data.to(device)
            output = model(data)
            y_pred = torch.softmax(output, dim=1)
            y_pred = torch.Tensor.cpu(y_pred)
            embedding = torch.Tensor.cpu(y_pred).detach().numpy()
            predict_list = np.concatenate((predict_list, embedding))
    predict_list = predict_list[1:]

    return predict_list
