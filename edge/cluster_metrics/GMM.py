import numpy as np
from sklearn.mixture import GaussianMixture


def GMM_metrics(x, select_num):
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
