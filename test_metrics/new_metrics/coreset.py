import pickle

from test_metrics.new_metrics.BALD import get_predict_list
import numpy as np
from datetime import datetime

from test_metrics.new_metrics.utils import get_sols


def coreset_selection(model, retrain_loader, select_size, lb, total_num):
    embedding = get_predict_list(retrain_loader, model)
    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(total_num, 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    print(datetime.now() - t_start)

    print('calculate greedy solution')
    t_start = datetime.now()
    lb_flag = lb.copy()
    mat = dist_mat[~lb_flag, :][:, lb_flag]
    for i in range(select_size):
        if i % 10 == 0:
            print('greedy solution {}/{}'.format(i, total_num))

    print(datetime.now() - t_start)
    lb_flag_ = lb.copy()
    subset = np.where(lb_flag_ == True)[0].tolist()
    SEED = 5
    r_name = 'mip{}.pkl'.format(SEED)
    w_name = 'sols.pkl'.format(SEED)
    sols = pickle.load(open('sols{}.pkl'.format(SEED), 'rb'))
    if sols is None:
        q_idxs = lb_flag
    else:
        lb_flag_[sols] = True
        q_idxs = lb_flag_
    print('sum q_idxs = {}'.format(q_idxs.sum()))
    select_index = np.arange(total_num)[(lb ^ q_idxs)]
    return select_index