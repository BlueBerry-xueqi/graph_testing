import numpy as np
from sklearn.cluster import MiniBatchKMeans


def MiniBatchKMeans_metrics(x, select_num):
    MBK = MiniBatchKMeans(n_clusters=select_num)
    MBK.fit(x)
    center_list = MBK.cluster_centers_
    select_index = []
    for p in center_list:
        distances_list = []
        for i in range(len(x)):
            distances = np.linalg.norm(x[i] - p)
            distances_list.append(distances)
        index = np.argsort(distances_list)[:1][0]
        select_index.append(index)
    return select_index


