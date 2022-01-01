from test_metrics.new_metrics.BALD import get_predict_list
import numpy as np
from datetime import datetime


def k_center_greedy_selection(model, retrain_loader, select_size, total_num, lb, cora_data, is_Cora, retrain_index, is_mem):
    embedding = get_predict_list(retrain_loader, model, is_mem, cora_data, is_Cora, retrain_index)
    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(total_num, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()

    print(datetime.now() - t_start)
    lb_flag = lb.copy()
    for i in range(select_size):
        if i % 10 == 0:
            print('greedy solution {}/{}'.format(i, select_size))
    select_index = np.arange(total_num)[(lb ^ lb_flag)]
    return select_index
