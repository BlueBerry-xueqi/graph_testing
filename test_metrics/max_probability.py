import numpy as np
import torch

def max_score(model, target_data):
    pre_prob, pre_labels, ground_truth = model.predict(target_data)
    pre_prob = np.exp(pre_prob)
    maxP, _ = torch.max( pre_prob, 1)
    return maxP, pre_labels, ground_truth 
