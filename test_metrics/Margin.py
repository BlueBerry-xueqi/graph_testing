import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def margin_score(model, target_loader, select_num):
    scores = []
    model.eval()
    with torch.no_grad():
        for data in target_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            y_pred = torch.softmax(output, dim=1)
            output_sort = np.sort(torch.Tensor.cpu(y_pred).detach().numpy())
            margin_score = output_sort[:, -1] - output_sort[:, -2]
            scores = np.concatenate((scores, margin_score))
     # sort through ascending order
    scores_sort = np.argsort(scores)
    select_index = scores_sort[:select_num]

    return select_index