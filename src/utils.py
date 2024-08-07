import math
import numpy as np
import scipy as sp
import torch.nn.functional as F
import networkx as nx
import torch
import random
from texttable import Texttable
from torch_geometric.utils import (
    erdos_renyi_graph,
    to_undirected,
    to_networkx,
    to_dense_adj,
    to_scipy_sparse_matrix,
    degree,
    get_laplacian,
    get_self_loop_attr
)
from torch_geometric.data import Data
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh

from torch_sparse import SparseTensor


def random_walk_positional_encoding(g, walk_length):
    num_nodes = g.num_nodes
    edge_index = g.edge_index

    adj = SparseTensor.from_edge_index(edge_index, None, sparse_sizes=(num_nodes, num_nodes))

    # Compute D^{-1} A:
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)

    out = adj
    row, col, value = out.coo()
    pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
    for _ in range(walk_length - 1):
        out = out @ adj
        row, col, value = out.coo()
        pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
    g['pos_enc'] = torch.stack(pe_list, dim=-1)
    return g

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    print(t.draw())

def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    return rank_corr_function(prediction, target, nan_policy='omit').correlation

def calculate_prec_at_k(k, prediction, target):
    """
    Calculating precision at k.
    """

    # increase k in case same similarity score values of k-th, (k+i)-th elements
    target_increase = np.sort(target)[::-1]
    target_value_sel = (target_increase >= target_increase[k - 1]).sum()
    target_k = max(k, target_value_sel)

    best_k_pred = prediction.argsort()[::-1][:k]
    best_k_target = target.argsort()[::-1][:target_k]

    return len(set(best_k_pred).intersection(set(best_k_target))) / k

def calculate_prec_at_k_dist(k, prediction, target):
    """
    Case distance:
    Calculating precision at k.
    """

    target_increase = np.sort(target)
    target_value_sel = (target_increase <= target_increase[k - 1]).sum()
    target_k = max(k, target_value_sel)

    best_k_pred = prediction.argsort()[:k]
    best_k_target = target.argsort()[:target_k]


    return len(set(best_k_pred).intersection(set(best_k_target))) / k

def normalize_sim_score_batch(g1_num_nodes, g2_num_nodes, sim_score):
    return torch.exp(-2 * (sim_score) / (g1_num_nodes + g2_num_nodes))  # For better performance.

def normalize_mcs_batch(g1_num_nodes, g2_num_nodes, mcs_val):
    return 2 * mcs_val / (g1_num_nodes + g2_num_nodes)  # For better performance.

# fmt: off
def aids_labels(g):
    types = [
        "O", "S", "C", "N", "Cl", "Br", "B", "Si", "Hg", "I", "Bi", "P", "F",
        "Cu", "Ho", "Pd", "Ru", "Pt", "Sn", "Li", "Ga", "Tb", "As", "Co", "Pb",
        "Sb", "Se", "Ni", "Te"
    ]

    return [types[i] for i in g.x.argmax(dim=1).tolist()]
# fmt: on

def draw_graphs(glist, aids=False):
    for i, g in enumerate(glist):
        plt.clf()
        G = to_networkx(g).to_undirected()
        if aids:
            label_list = aids_labels(g)
            labels = {}
            for j, node in enumerate(G.nodes()):
                labels[node] = label_list[j]
            nx.draw(G, labels=labels)
        else:
            nx.draw(G)
        plt.savefig("graph{}.png".format(i))

def draw_weighted_nodes(filename, g, model):
    """
    Draw graph with weighted nodes (for AIDS).
    """
    features = model.convolutional_pass(g.edge_index, g.x)
    coefs = model.attention.get_coefs(features)

    print(coefs)

    plt.clf()
    G = to_networkx(g).to_undirected()

    label_list = aids_labels(g)
    labels = {}
    for i, node in enumerate(G.nodes()):
        labels[node] = label_list[i]

    vmin = coefs.min().item() - 0.005
    vmax = coefs.max().item() + 0.005

    nx.draw(
        G,
        node_color=coefs.tolist(),
        cmap=plt.cm.Reds,
        labels=labels,
        vmin=vmin,
        vmax=vmax,
    )

    # sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm.set_array(coefs.tolist())
    # cbar = plt.colorbar(sm)

    plt.savefig(filename)
