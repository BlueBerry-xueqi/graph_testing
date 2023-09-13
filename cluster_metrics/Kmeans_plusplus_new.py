import math
import numpy as np
import torch
from sklearn.cluster import kmeans_plusplus

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


def KMeans_plusplus_score(model, retrain_index, data, select_num):
    with torch.no_grad():
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        predict_list = torch.Tensor.cpu(y_pred).detach().numpy()

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