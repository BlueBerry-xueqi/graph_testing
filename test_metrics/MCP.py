#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import argparse
# import condition
import numpy as np

CLIP_MAX = 0.5


def find_second(act, ncl=10):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(ncl):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(ncl):
        if i == max_index:
            continue
        if act[i] > second_max:
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio


# for wilds datasets
def select_wilds_only(model, selectsize, x_target, ncl):
    window_size = int(ncl * ncl)
    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(window_size)]
    dicindex = [[] for i in range(window_size)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act, ncl)  # max_index
        dicratio[max_index * ncl + sec_index].append(ratio)
        dicindex[max_index * ncl + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    selected_idx = []
    for i in range(selectsize):
        selected_idx.append(selected_lst[i])

    return selected_idx


def select_from_firstsec_dic(selectsize, dicratio, dicindex, ncl=10):
    selected_lst = []
    tmpsize = selectsize

    noempty = no_empty_number(dicratio)
    # print(selectsize)
    # print(noempty)
    window_size = int(ncl * ncl)
    while selectsize >= noempty:
        for i in range(window_size):
            if len(dicratio[i]) != 0:
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                # if tmp>=0.1:
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)
        # print(selectsize)
    # selectsize<noempty
    # no_empty_number(dicratio)
    # print(selectsize)

    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]  # 剩下多少就申请多少
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(window_size):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    # selected_lst.append()
                    # if tmp_max>=0.1:
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]  # 吧样本序列号存在此列表中
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
        print(len(selected_lst))
    # print(selected_lst)
    assert len(selected_lst) == tmpsize
    return selected_lst


def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty


def find_index(target_lsa, selected_lst, max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa == target_lsa[i] and i not in selected_lst:
            return i
    return 0


def order_output(target_lsa, select_amount):
    lsa_lst = []

    tmp_lsa_lst = target_lsa[:]
    selected_lst = []
    while len(selected_lst) < select_amount:
        max_lsa = max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa, selected_lst, max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst, lsa_lst


def fetch_our_measure(model, x_target):
    bound_data_lst = []
    # x_test=x_test.astype('float32').reshape(-1,28,28,1)
    # x_test/=255
    act_layers = model.predict(x_target)

    ratio_lst = []
    for i in range(len(act_layers)):
        act = act_layers[i]
        _, __, ratio = find_second(act)
        ratio_lst.append(ratio)

    return ratio_lst


def MCP_selection_wilds(model, target_data, select_size, ncl):
    print("Prepare...")
    select_index = select_wilds_only(model, select_size, target_data, ncl)
    print("Over...")
    return select_index


def MCP_score(model, target_data, select_num, ncl):
    """

    Args:
        model:
        target_data:a
        ncl: number of class

    Returns:

    """
    select_size = len(target_data)
    select_index = MCP_selection_wilds(model, target_data, select_size, ncl)
    select_index = select_index[-select_num:]
    return select_index


def margin_score(model, target_data):
    prediction, pre_labels, ground_truth = model.predict(target_data)
    prediction_sorted = np.sort(prediction)
    margin_list = prediction_sorted[:, -1] / prediction_sorted[:, -2]
    return margin_list, pre_labels, ground_truth
