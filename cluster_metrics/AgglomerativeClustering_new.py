import pdb
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

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


def AgglomerativeClustering_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    AC = AgglomerativeClustering(n_clusters=10)
    AC.fit(x)
    label_list = list(AC.labels_)
    n = int(select_num / 10)
    select_index = predict(x, label_list, n)
    return select_index
