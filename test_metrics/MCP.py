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
<<<<<<< HEAD

CLIP_MAX = 0.5


def find_second(act, ncl=10):
=======
CLIP_MAX = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_second(act, matrix_size):
>>>>>>> 3533dc3257a775f78a7faee15381d0bbe036c694
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(matrix_size):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(matrix_size):
        if i == max_index:
            continue
        if act[i] > second_max:
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio


<<<<<<< HEAD
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
=======
# 配对表情非空的数目。比如第一是3，第二是5，此时里面没有任何实例存在那么就是0
def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty


# 输入第一第二大的字典，输出selected_lst。用例的index
def select_from_firstsec_dic(selectsize, dicratio, dicindex, ms):
>>>>>>> 3533dc3257a775f78a7faee15381d0bbe036c694
    selected_lst = []
    tmpsize = selectsize

    noempty = no_empty_number(dicratio)
<<<<<<< HEAD
    # print(selectsize)
    # print(noempty)
    window_size = int(ncl * ncl)
    while selectsize >= noempty:
        for i in range(window_size):
            if len(dicratio[i]) != 0:
=======
    # 待选择的数目大于非空的类别数(满载90类)，每一个都选一个
    while selectsize >= noempty:
        for i in range(ms):
            if len(dicratio[i]) != 0:  # 非空就选一个最大的出来
>>>>>>> 3533dc3257a775f78a7faee15381d0bbe036c694
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)

    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]  # 剩下多少就申请多少
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(ms):
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
    assert len(selected_lst) == tmpsize
    return selected_lst


<<<<<<< HEAD
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
=======
def mcp_score(model, target_loader, num, class_num):
    dicratio = [[] for i in range(class_num * class_num)]  # 只用90，闲置10个
    dicindex = [[] for i in range(class_num * class_num)]
    model.eval()
    with torch.no_grad():
        for batch_no, (x_batch, _) in enumerate(target_loader):
            x_batch = x_batch.to(device)
            output = model(x_batch)
            output_copy = funSoft(torch.Tensor.cpu(output)).detach().numpy()
            for i in range(len(output_copy)):
                index_in_data = batch_size * batch_no + i
                act = output_copy[i]
                max_index, sec_index, ratio = find_second(act, class_num)
                # 安装第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号
                dicratio[max_index * class_num + sec_index].append(ratio)
                dicindex[max_index * class_num + sec_index].append(index_in_data)
    mcp_list = select_from_firstsec_dic(num, dicratio, dicindex, class_num * class_num)
    select_index = mcp_list[:num]
    
    return select_index

>>>>>>> 3533dc3257a775f78a7faee15381d0bbe036c694
