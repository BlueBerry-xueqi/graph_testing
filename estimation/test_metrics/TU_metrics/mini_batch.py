import pdb
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MiniBatchKMeans_metrics(model, retrain_index, data , select_num):
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        x = torch.Tensor.cpu(y_pred).detach().numpy()
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
