import pdb
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from test_metrics.util.predict_TU import get_predict_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(x, label_list):
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
            distances = np.linalg.norm(x[i] - p)
            distances_list.append(distances)
        index = np.argsort(distances_list)[:1][0]
        index_list.append(index)
    return index_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def SpectralClustering_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    SC = SpectralClustering(n_clusters=select_num, assign_labels='discretize', random_state=0)

    SC.fit(x)
    label_list = list(SC.labels_)
    select_index = predict(x, label_list)
    return select_index