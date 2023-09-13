import pdb
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from test_metrics.util.predict_TU import get_predict_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MiniBatchKMeans_metrics(model, retrain_loader , select_num):
    x = get_predict_list(retrain_loader, model)
    MBK = MiniBatchKMeans(n_clusters=select_num, random_state=0)
    MBK.fit(x)
    center_list = MBK.cluster_centers_
    select_index = []
    for p in center_list:
        distances_list = []
        for i in range(len(x)):
            distances = np.linalg.norm(x[i] - p)
            distances_list.append(distances)
        index_center_list = list(np.argsort(distances_list)[:1])
        select_index += index_center_list

    return select_index
