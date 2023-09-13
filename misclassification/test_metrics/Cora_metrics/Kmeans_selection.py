import math

import torch
from sklearn.cluster import KMeans

from test_metrics.util.predict_list_mnist import get_predict_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def KMeans_score(model, retrain_index, data, select_num):
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        predict_list = torch.Tensor.cpu(y_pred).detach().numpy()
        
    kmeans = KMeans(n_clusters=select_num, random_state=0).fit(predict_list)
    centers = kmeans.cluster_centers_
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
