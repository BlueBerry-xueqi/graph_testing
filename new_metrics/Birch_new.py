import pdb
import numpy as np
import torch
from sklearn.cluster import Birch
from test_metrics.TU_metrics.BALD import get_predict_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Birch_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    brc = Birch(n_clusters=select_num)
    brc.fit(x)

    center_list = brc.subcluster_centers_
    if len(brc.subcluster_centers_) < select_num:
        number_ap = select_num - len(brc.subcluster_centers_)
        tmp_center = center_list[0]
        for i in range(number_ap):
            center_list = np.insert(center_list, 1, tmp_center, axis=0)

    select_index = []
    for p in center_list:
        distances_list = []
        for i in range(len(x)):
            distances = np.linalg.norm(x[i] - p)
            distances_list.append(distances)
        index = np.argsort(distances_list)[:1][0]
        select_index.append(index)

    return select_index