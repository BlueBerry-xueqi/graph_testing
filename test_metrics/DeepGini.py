import numpy as np

def deepgini_score(model, target_data):
    prediction = model.predict(target_data)
    gini_list = np.sum(prediction ** 2, axis=1)
    return gini_list

