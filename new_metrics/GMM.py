import pdb
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from test_metrics.TU_metrics.BALD import get_predict_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

