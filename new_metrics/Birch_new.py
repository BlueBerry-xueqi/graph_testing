import pdb
import numpy as np
import torch
from sklearn.cluster import Birch
from test_metrics.TU_metrics.BALD import get_predict_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Birch_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    brc = Birch(n_clusters=10)
    brc.fit(x)

    center_list = brc.subcluster_centers_
    if len(brc.subcluster_centers_) < 10:
        number_ap = 10 - len(brc.subcluster_centers_)
        tmp_center = center_list[0]
        for i in range(number_ap):
            center_list = np.insert(center_list, 1, tmp_center, axis=0)

    select_index = []
    n = int(select_num / 10)
    for p in center_list:
        distances_list = []
        for i in range(len(x)):
            distances = np.linalg.norm(x[i] - p)
            distances_list.append(distances)
        index_center_list = list(np.argsort(distances_list)[:n])
        select_index += index_center_list

    return select_index