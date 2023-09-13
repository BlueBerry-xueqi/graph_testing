import numpy as np
import random
import networkx as nx


def Random_rank_idx(x):
    random_rank_idx = random.sample(range(0, len(x)), len(x))
    return random_rank_idx


def Degree_rank_idx(G, candidate_idx):
    dic = {}
    for i in candidate_idx:
        try:
            dic[i] = G.degree(i)
        except:
            dic[i] = 0
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    degree_rank_idx = np.array([i[0] for i in L])
    return degree_rank_idx


def Eccentricity_rank_idx(G, candidate_idx):

    dic_all = nx.eccentricity(G)
    dic = {}
    for i in candidate_idx:
        if i in dic_all:
            dic[i] = dic_all[i]
        else:
            dic[i] = 0

    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    eccentricity_rank_idx = np.array([i[0] for i in L][::-1])
    return eccentricity_rank_idx


def Center_rank_idx(G, candidate_idx):
    center = nx.center(G)[0]
    dic_all = nx.single_source_shortest_path_length(G, center)
    dic ={}
    for i in candidate_idx:
        if i in dic_all:
            dic[i] = dic_all[i]
        else:
            dic[i] = 0
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    center_rank_idx = np.array([i[0] for i in L][::-1])
    return center_rank_idx


def Betweenness_Centrality_rank_idx(G, candidate_idx):
    dic_all = nx.betweenness_centrality(G)
    dic ={}
    for i in candidate_idx:
        if i in dic_all:
            dic[i] = dic_all[i]
        else:
            dic[i] = 0
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    bc_rank_idx = np.array([i[0] for i in L])
    return bc_rank_idx


def Eigenvector_Centrality_rank_idx(G, candidate_idx):
    dic_all = nx.eigenvector_centrality(G)
    dic ={}
    for i in candidate_idx:
        if i in dic_all:
            dic[i] = dic_all[i]
        else:
            dic[i] = 0
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    ec_rank_idx = np.array([i[0] for i in L])
    return ec_rank_idx


def PageRank_rank_idx(G, candidate_idx):
    dic_all = nx.pagerank(G, alpha=0.85)
    dic ={}
    for i in candidate_idx:
        if i in dic_all:
            dic[i] = dic_all[i]
        else:
            dic[i] = 0
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    pr_rank_idx = np.array([i[0] for i in L])
    return pr_rank_idx


def Hits_rank_idx(G, candidate_idx):
    dic_all, _ = nx.hits(G, max_iter=50)
    dic ={}
    for i in candidate_idx:
        if i in dic_all:
            dic[i] = dic_all[i]
        else:
            dic[i] = 0
    L = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    hits_rank_idx = np.array([i[0] for i in L])
    return hits_rank_idx

