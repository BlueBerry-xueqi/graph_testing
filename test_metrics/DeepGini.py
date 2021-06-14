import numpy as np
import torch

def deepgini_score(model, target_data):
    pre_prob, pre_labels, ground_truth = model.predict(target_data)
    pre_prob = np.exp(pre_prob)
    gini_list = 1 - np.sum(pre_prob ** 2, axis=1)
    return gini_list, pre_labels, ground_truth 



