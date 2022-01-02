import pdb
import math
import pandas as pd
import numpy as np
import torch

from test_metrics.TU_metrics.BALD import get_predict_list


def euler_distance(point1: np.ndarray, point2: list) -> float:
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


class ClusterNode(object):
    def __init__(self, vec, left=None, right=None, distance=-1, id=None, count=1):
        self.vec = vec
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id
        self.count = count


class Hierarchical(object):

    def __init__(self, k = 1):
        assert k > 0
        self.k = k
        self.labels = None

    def fit(self, x):
        nodes = [ClusterNode(vec=v, id=i) for i,v in enumerate(x)]
        distances = {}
        point_num, future_num = np.shape(x)
        self.labels = [ -1 ] * point_num
        currentclustid = -1
        while len(nodes) > self.k:
            min_dist = math.inf
            nodes_len = len(nodes)
            closest_part = None
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    d_key = (nodes[i].id, nodes[j].id)
                    if d_key not in distances:
                        distances[d_key] = euler_distance(nodes[i].vec, nodes[j].vec)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_vec = [ (node1.vec[i] * node1.count + node2.vec[i] * node2.count ) / (node1.count + node2.count)
                        for i in range(future_num)]
            new_node = ClusterNode(vec=new_vec,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   id=currentclustid,
                                   count=node1.count + node2.count)
            currentclustid -= 1
            del nodes[part2], nodes[part1]
            nodes.append(new_node)
        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        for i, node in enumerate(self.nodes):
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        if node.left == None and node.right == None:
            self.labels[node.id] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)

    def predict(self, x, label_list):
        df = pd.DataFrame(x)
        df['label'] = label_list
        center_list = []
        for i in list(set(label_list)):
            tmp_df = df[df['label'] == i].copy()
            del tmp_df["label"]
            tmp_x = tmp_df.to_numpy()
            center = np.average(tmp_x, axis=0)
            center_list.append(center)
        index_list = []
        for p in center_list:
            distances_list = []
            for i in range(len(x)):
                distances = np.linalg.norm(x[i] - p)
                distances_list.append(distances)
            index = np.argsort(distances_list)[:1][0]
            index_list.append(index)
        return index_list


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Hierarchical_metrics(model, retrain_loader, select_num):
    x = get_predict_list(retrain_loader, model)
    hierarchical = Hierarchical(k=select_num)
    hierarchical.fit(x)
    label_list = hierarchical.labels
    select_index = hierarchical.predict(x, label_list)
    return select_index

