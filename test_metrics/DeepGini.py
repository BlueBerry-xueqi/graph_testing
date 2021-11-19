import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def deepgini_score(model, target_loader, select_num):
    # pre_prob, pre_labels, ground_truth = model.predict(target_data)
    scores = []
    with torch.no_grad():
        for data in target_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            y_pred = torch.softmax(output, dim=1)
            gini_score = 1 - np.sum(np.power(np.exp(torch.Tensor.cpu(y_pred).detach().numpy()), 2), axis=1)
            scores = np.concatenate((scores, gini_score))
    # sort through ascending order
    scores_sort = np.argsort(scores)
    select_index = scores_sort[-select_num:]
    print("==========select_index============")
    return select_index
