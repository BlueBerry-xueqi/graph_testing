import numpy as np
import torch
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def BALD_metrics(model, retrain_index, data, select_num):
    BALD_list = []
    mode_list = []
    data_len = len(retrain_index)
    print("Prepare...")
    for _ in range(20):
        data = data.to(device)
        output = model(data)[retrain_index]
        y_pred = torch.softmax(output, dim=1)
        prediction = torch.Tensor.cpu(y_pred).detach().numpy()
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1, ))[1][0]
        mode_list.append(1 - mode_num / 50)

    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[-select_num:]
    return select_index
