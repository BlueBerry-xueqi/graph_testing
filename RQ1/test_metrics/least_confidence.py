import pdb
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def least_confidence_score(model, retrain_index, select_num, data_cora, is_Cora, retrain_loader, is_mem):
    scores = []
    with torch.no_grad():
        if is_Cora:
            data_cora = data_cora.to(device)
            output = model(data_cora)[retrain_index]
            y_pred = torch.softmax(output, dim=1)
            score = np.max(torch.Tensor.cpu(y_pred).detach().numpy(), axis=1)
            scores = np.concatenate((scores, score))
        else:
            for data in retrain_loader:
                data = data.to(device)
                if is_mem:
                    output = model(data)[0]
                else:
                    output = model(data)
                prediction = torch.softmax(output, dim=1).numpy()
                max_pre = prediction.max(1)
                scores = np.concatenate((scores, max_pre))

    select_index = np.argsort(scores)[::-1][:select_num]
    select_index = select_index.copy()
    return select_index
