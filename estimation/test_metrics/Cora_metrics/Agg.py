import pdb
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(x, label_list, n):
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
        index_center_list = list(np.argsort(distances_list)[:n])
        index_list += index_center_list
    return index_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def AgglomerativeClustering_metrics(model, retrain_index, data, select_num):
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        predict_list = torch.Tensor.cpu(y_pred).detach().numpy()

    x = predict_list
    AC = AgglomerativeClustering(n_clusters=select_num)
    AC.fit(x)
    label_list = list(AC.labels_)
    # n = int(select_num / 10)
    select_index = predict(x, label_list, 1)
    return select_index