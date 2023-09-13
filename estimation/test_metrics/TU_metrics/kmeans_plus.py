import math

import torch
from sklearn.cluster import kmeans_plusplus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from test_metrics.util.predict_TU import get_predict_list


def KMeans_plusplus_score(model, retrain_loader, select_num):
    predict_list = get_predict_list(retrain_loader, model)

    centers, _ = kmeans_plusplus(X=predict_list, n_clusters=select_num, random_state=0)

    selected_index = predict(predict_list, centers)
    return selected_index


def predict(pred_list, centers):
    selected_index = []
    for center in centers:
        min_distance = 100
        for i in range(0, len(pred_list)):
            pred = pred_list[i]
            distance = eucliDist(pred, center)
            if distance < min_distance:
                min_distance = distance
                select_index = i
        selected_index.append(select_index)
    return selected_index


def eucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))