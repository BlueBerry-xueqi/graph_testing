import pdb

from scipy.stats import entropy
import numpy as np
from scipy import stats
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_predict_list(retrain_loader, model, is_mem):
    predict_list = np.array([[0, 0]])
    for data in retrain_loader:
        data = data.to(device)
        if is_mem:
            output = model(data)[0]
        else:
            output = model(data)
        y_pred = torch.softmax(output, dim=1)
        y_pred = torch.Tensor.cpu(y_pred)
        embedding = y_pred.detach().numpy()
        predict_list = np.concatenate((predict_list, embedding))
    predict_list = predict_list[1:]
    return predict_list


# import tensorflow.keras.backend as K
def BALD_selection(model, retrain_loader, select_size, total_size, is_mem, cora_data, is_Cora, retrain_index):
    BALD_list = []
    mode_list = []
    data_len = total_size
    print("Prepare...")
    for _ in range(20):
        prediction = get_predict_list(retrain_loader, model, is_mem, cora_data, is_Cora, retrain_index)
        BALD_list.append(prediction)
    BALD_list = np.asarray(BALD_list)
    for _ in range(data_len):
        mode_num = stats.mode(BALD_list[:, _:(_ + 1), ].reshape(-1, ))[1][0]

        mode_list.append(1 - mode_num / 50)

    sorted_index = np.argsort(mode_list)
    select_index = sorted_index[-(select_size):]
    return sorted_index
