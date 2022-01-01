import pdb

import numpy as np
import torch

from test_metrics.TU_metrics.BALD import get_predict_list


class K_Means(object):
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data, dic):
        index_list = []
        for key in dic:
            center = dic[key]
            distances_list = []
            for point in p_data:
                distances = np.linalg.norm(point - center)
                distances_list.append(distances)
            index = np.argsort(distances_list)[:1][0]
            index_list.append(index)
        return index_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Kmeans_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    k_means = K_Means(k=select_num)
    k_means.fit(x)
    dic = k_means.centers_
    select_index = k_means.predict(x, dic)
    return select_index
