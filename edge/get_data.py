import pandas as pd
import networkx as nx
import numpy as np
import pickle
import math
from tdc.multi_pred import DDI, PPI, GDA, DTI
from sklearn.feature_extraction.text import TfidfVectorizer


def edge_index_to_adj(edge_index_np):
    n_node = max(edge_index_np[0])+1
    m = np.full((n_node, n_node), 0)
    i_j_list = []
    for idx in range(len(edge_index_np[0])):
        i = edge_index_np[0][idx]
        j = edge_index_np[1][idx]
        if [i, j] not in i_j_list and [j, i] not in i_j_list:
            i_j_list.append([i, j])

    for v in i_j_list:
        i = v[0]
        j = v[1]
        m[i][j] = 1
        m[j][i] = 1
    return m


def get_DrugBank():
    """86 edge classes"""
    data = DDI(name='DrugBank')
    dataset = data.get_data()

    y_list = []
    left_node_list = []
    right_node_list = []

    for i, row in dataset.iterrows():
        # drug1 = row['Drug1_ID']
        # drug2 = row['Drug2_ID']
        drug1 = row['Drug1']
        drug2 = row['Drug2']
        left_node_list.append(drug1)
        right_node_list.append(drug2)
        y = row['Y']
        y_list.append(y)

    node_list = sorted(list(set(left_node_list+right_node_list)))

    vectorizer = TfidfVectorizer(max_features=64)
    x_np = vectorizer.fit_transform(node_list).toarray()
    pickle.dump(x_np, open('./data/DrugBank/x_np.pkl', 'wb'))

    dic = dict(zip(node_list, range(len(node_list))))
    left_node_list = [dic[i] for i in left_node_list]
    right_node_list = [dic[i] for i in right_node_list]

    df = pd.DataFrame(columns=['left', 'right', 'label'])
    df['left'] = left_node_list
    df['right'] = right_node_list
    df['label'] = y_list
    df = df.sort_values(by=['left']).reset_index(drop=True)

    left_list = list(df['left'])
    right_list = list(df['right'])

    edge_index_np = [left_list, right_list]
    edge_index_np = np.array(edge_index_np)

    unique_label = sorted(list(set(y_list)))
    dic_label = dict(zip(unique_label, range(len(unique_label))))
    y_np = [dic_label[i] for i in y_list]
    y_np = np.array(y_np)

    pickle.dump(edge_index_np, open('./data/DrugBank/edge_index_np.pkl', 'wb'))
    pickle.dump(y_np, open('./data/DrugBank/y_np.pkl', 'wb'))


def get_BindingDB():
    """6 edge classes"""
    data = DTI(name='BindingDB_Kd')
    dataset = data.get_data()

    y_list = []
    left_node_list = []
    right_node_list = []

    for i, row in dataset.iterrows():

        drug1 = row['Drug']
        drug2 = row['Target']

        left_node_list.append(drug1)
        right_node_list.append(drug2)
        y = row['Y']
        y_list.append(y)
    node_list = sorted(list(set(left_node_list+right_node_list)))
    unique_label = sorted(list(set(y_list)))

    re_y_list = []
    p_10 = 1
    p_20 = 100
    p_30 = 1000
    p_40 = 10000
    p_50 = 100000

    for i in y_list:
        if i<=p_10:
            re_y_list.append(0)
        elif p_10<i<=p_20:
            re_y_list.append(1)
        elif p_20<i<=p_30:
            re_y_list.append(2)
        elif p_30<i<=p_40:
            re_y_list.append(3)
        elif p_40<i<=p_50:
            re_y_list.append(4)
        else:
            re_y_list.append(5)

    y_list = re_y_list
    print(set(y_list))

    vectorizer = TfidfVectorizer(max_features=64)
    x_np = vectorizer.fit_transform(node_list).toarray()
    pickle.dump(x_np, open('./data/BindingDB/x_np.pkl', 'wb'))

    dic = dict(zip(node_list, range(len(node_list))))
    left_node_list = [dic[i] for i in left_node_list]
    right_node_list = [dic[i] for i in right_node_list]

    df = pd.DataFrame(columns=['left', 'right', 'label'])
    df['left'] = left_node_list
    df['right'] = right_node_list
    df['label'] = y_list
    df = df.sort_values(by=['left']).reset_index(drop=True)

    left_list = list(df['left'])
    right_list = list(df['right'])

    edge_index_np = [left_list, right_list]
    edge_index_np = np.array(edge_index_np)

    unique_label = sorted(list(set(y_list)))
    dic_label = dict(zip(unique_label, range(len(unique_label))))
    y_np = [dic_label[i] for i in y_list]
    y_np = np.array(y_np)

    pickle.dump(edge_index_np, open('./data/BindingDB/edge_index_np.pkl', 'wb'))
    pickle.dump(y_np, open('./data/BindingDB/y_np.pkl', 'wb'))


if __name__ == '__main__':
    get_DrugBank()
    get_BindingDB()


