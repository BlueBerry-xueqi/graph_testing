import pdb
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from test_metrics.TU_metrics.BALD import get_predict_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(x, label_list, n):
    """
    Args:
        x: <numpy>, matrix
        label_list: <list>
        n: <int>, The number of getting each center point.

    Returns: <list>, index list.
    """
    df = pd.DataFrame(x)
    df['label'] = label_list
    center_list = []
    for i in list(set(label_list)):
        tmp_df = df[df['label'] == i].copy()
        del tmp_df["label"]
        tmp_x = tmp_df.to_numpy()
        center = np.average(tmp_x, axis=0)
        center_list.append(center)
    index_list = []
    for p in center_list:
        distances_list = []
        for i in range(len(x)):
            # distances = np.linalg.norm(x[i] - p)
            distances = np.sqrt(np.sum(np.square(x[i] - p)))
            distances_list.append(distances)
        index_center_list = list(np.argsort(distances_list)[:n])
        index_list += index_center_list
    return index_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SpectralClustering_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    SC = SpectralClustering(n_clusters=10, assign_labels='discretize', random_state=0)

    SC.fit(x)
    label_list = list(SC.labels_)
    n = int(select_num / 10)
    select_index = predict(x, label_list, n)
    return select_index