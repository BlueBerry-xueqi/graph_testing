import pdb
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from test_metrics.util.predict_list_mnist import get_predict_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def GMM_metrics(model, retrain_index,data, select_num):
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        x = torch.Tensor.cpu(y_pred).detach().numpy()
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
