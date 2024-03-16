import math
import numpy as np
from sklearn.cluster import kmeans_plusplus, KMeans


def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def predict(pred_list, centers):
    selected_index_list = []
    for center in centers:
        min_distance = 100
        for i in range(0, len(pred_list)):
            pred = pred_list[i]
            distance = eucliDist(pred, center)
            if distance < min_distance:
                min_distance = distance
                select_index = i
        selected_index_list.append(select_index)
    return selected_index_list


def KMeans_plusplus_metrics(x, select_num):

    centers, _ = kmeans_plusplus(X=x, n_clusters=select_num)
    selected_index = predict(x, centers)
    return selected_index


def KMeans_metrics(x, select_num):
    kmeans = KMeans(n_clusters=select_num).fit(x)
    centers = kmeans.cluster_centers_
    selected_index = predict(x, centers)
    return selected_index




