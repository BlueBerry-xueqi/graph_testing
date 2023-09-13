import pdb
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predict_list(retrain_loader, model):
    predict_list = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
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


def GMM_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    gm = GaussianMixture(n_components=select_num, random_state=0)
    gm.fit(x)
    center_list = gm.means_
    select_index = []
    for p in center_list:
        distances_list = []
        for i in range(len(x)):
            distances = np.linalg.norm(x[i] - p)
            distances_list.append(distances)
        index = np.argsort(distances_list)[:1][0]
        select_index.append(index)

    return select_index

