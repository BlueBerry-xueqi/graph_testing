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
        if act[i] > second_max:  # 第2大加一个限制条件，那就是不能和max_一样
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio  # ratio是第二大输出达到最大输出的百分比


# for wilds datasets
def select_wilds_only(model, selectsize, x_target, ncl):
    window_size = int(ncl * ncl)
    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(window_size)]  # 只用90，闲置10个
    dicindex = [[] for i in range(window_size)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act, ncl)  # max_index
        # 安装第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号
        dicratio[max_index * ncl + sec_index].append(ratio)
        dicindex[max_index * ncl + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    selected_idx = []
    for i in range(selectsize):
        selected_idx.append(selected_lst[i])

    return selected_idx



# 输入第一第二大的字典，输出selected_lst。用例的index
def select_from_firstsec_dic(selectsize, dicratio, dicindex, ncl=10):
    selected_lst = []
    tmpsize = selectsize
    # tmpsize保存的是采样大小，全程都不会变化

    noempty = no_empty_number(dicratio)
    # print(selectsize)
    # print(noempty)
    # 待选择的数目大于非空的类别数(满载90类)，每一个都选一个
    window_size = int(ncl * ncl)
    while selectsize >= noempty:
        for i in range(window_size):
            if len(dicratio[i]) != 0:  # 非空就选一个最大的出来
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

    # 剩下少量样本没有采样，比如还存在30类别非空，但是只要采样10个，此时我们取30个最大值中的前10大
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


# 配对表情非空的数目。比如第一是3，第二是5，此时里面没有任何实例存在那么就是0
def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty


# 找到前select_amount大的值的index输出
# 这个函数得修改一下

# 找出max_lsa在 target_lsa中的index，排除selected_lst中已经选的
def find_index(target_lsa, selected_lst, max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa == target_lsa[i] and i not in selected_lst:
            return i
    return 0


# 重新修改
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


def MCP_score(model, target_data, ncl):
    """

    Args:
        model:
        target_data:
        ncl: number of class

    Returns:

    """
    select_size = len(target_data)
    select_index = MCP_selection_wilds(model, target_data, select_size, ncl)
    total_score, _, _ = margin_score(model, target_data)
    ranked_score = total_score[select_index]
    return ranked_score


def margin_score(model, target_data):
    prediction,  pre_labels, ground_truth = model.predict(target_data)
    prediction_sorted = np.sort(prediction)
    margin_list = prediction_sorted[:, -1] / prediction_sorted[:, -2]
    return margin_list,   pre_labels, ground_truth

