import numpy as np
import torch
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predict_list(retrain_loader, model):
    predict_list = np.array([[0, 0]])
    with torch.no_grad():
        for data in retrain_loader:
            data = data.to(device)
            output = model(data)[0]
            y_pred = torch.softmax(output, dim=1)
            y_pred = torch.Tensor.cpu(y_pred)
            embedding = torch.Tensor.cpu(y_pred).detach().numpy()
            predict_list = np.concatenate((predict_list, embedding))
    predict_list = predict_list[1:]
    return predict_list


def BALD_metrics(model, retrain_loader, select_size, retrain_dataset_length):
    BALD_list = []
    mode_list = []
    data_len = retrain_dataset_length
    print("Prepare...")
    for _ in range(20):
        prediction = get_predict_list(retrain_loader, model)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1, ))[1][0]
        mode_list.append(1 - mode_num / 50)

    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[-(select_size):]
    return select_index
