import pdb

import numpy
import numpy as np
import random


# select data randomly
def random_select(data_length, select_num):
    selected_index = random.sample(np.arange(data_length).tolist(), select_num)

    return selected_index
