import pdb

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# training data with low scores will be selected for training.
def margin_score(model, retrain_index, select_num, data_cora, is_Cora, retrain_loader, is_mem):
    scores = []
    model.eval()
    with torch.no_grad():
        if is_Cora:
            data_cora = data_cora.to(device)
            output = model(data_cora)[retrain_index]
            y_pred = torch.softmax(output, dim=1)
            output_sort = np.sort(y_pred.detach().numpy())
            margin_score = output_sort[:, -1] - output_sort[:, -2]
            scores = np.concatenate((scores, margin_score))
        else:
            for data in retrain_loader:
                data = data.to(device)
                if is_mem:
                    output = model(data)[0]
                else:
                    output = model(data)
                y_pred = torch.softmax(output, dim=1)

                output_sort = np.sort(y_pred.detach().numpy())
                margin_score = output_sort[:, -1] - output_sort[:, -2]
                scores = np.concatenate((scores, margin_score))

    # sort through ascending order
    scores_sort = np.argsort(scores)
    select_index = scores_sort[:select_num]
    return select_index
