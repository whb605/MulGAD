import copy

import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import Counter
from collections import OrderedDict
import torch
import scipy.io as sio
import random
import dgl
from sklearn.metrics import precision_recall_curve, auc


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = np.where(rowsum == 0, 1, rowsum)  # 属性全为0
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return adj_normalized


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    print(dataset)
    data = sio.loadmat(dataset)
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = np.array(network.todense())

    matrix = np.identity(adj.shape[0], dtype=int)
    adj = np.bitwise_or(adj, matrix)
    adj = sp.csr_matrix(adj)
    feat = sp.lil_matrix(attr)
    labels = np.zeros((adj.shape[0], 2))

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[: num_train]
    idx_val = all_idx[num_train: num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.from_networkx(nx_graph)
    return dgl_graph


def generate_neighbor_nodes(dgl_graph, node_idx):
    neighbor_nodes = []
    for curr_node in node_idx:
        x = dgl_graph.successors(curr_node)
        perm = torch.randperm(x.size(0))
        x_shuffled = x[perm]
        neighbor_nodes.append(x_shuffled)
    return neighbor_nodes


def generate_rwr_subgraph(adj_list, neighbor_list, subgraph_size, features, subgraph_mask_adj):
    """Generate subgraph with RWR algorithm."""
    num_view = neighbor_list.size(0)
    num_node = neighbor_list.size(1)
    subgraphs = []
    subviews = []
    for curr_node in range(num_node):
        tmpg = []
        tmpv = []
        for curr_view in range(num_view):
            neighbors = neighbor_list[curr_view][curr_node]
            neighbor_idx = list(range(0, torch.sum(neighbors >= 0)))
            tmp_num = 0
            for idx in neighbor_idx:
                curr_neighbor = neighbors[idx]
                if curr_neighbor >= 0:
                    tmpg.append(curr_neighbor)
                    tmpv.append(curr_view)
                    tmp_num = tmp_num + 1
                if tmp_num >= subgraph_size[curr_view]: break
        subgraphs.append(tmpg)
        subviews.append(tmpv)

    ba = []
    bf = []
    subgraph_view_size = torch.sum(subgraph_size) + 1

    for i in range(len(subgraphs)):
        tmp_adj = adj_list[:, subgraphs[i], :][:, :, subgraphs[i]]
        view_c = 0
        view_n = Counter(subviews[i])

        cur_feat = torch.zeros(subgraph_view_size, features.shape[1], dtype=torch.float).cuda()
        cur_adj = torch.zeros(subgraph_view_size, subgraph_view_size, dtype=torch.int).cuda()
        cur_adj[0, 0] = 1
        for value, count in view_n.items():
            cur_begin = torch.sum(subgraph_size[0:value]) + 1
            cur_feat[cur_begin:cur_begin + count, :] = features[subgraphs[i][view_c:view_c + count], :]
            cur_adj[cur_begin:cur_begin + count, cur_begin:cur_begin + count] = tmp_adj[value, view_c:view_c + count,
                                                                                view_c:view_c + count]
            view_c = view_c + count
            cur_adj[0, cur_begin] = 1
            cur_adj[cur_begin, 0] = 1
        cur_feat[0] = cur_feat[1]
        cur_adj = cur_adj * subgraph_mask_adj
        cur_adj = cur_adj.to(torch.float)
        cur_adj = normalize_adj(cur_adj)
        ba.append(torch.unsqueeze(cur_adj, 0))
        bf.append(torch.unsqueeze(cur_feat, 0))

    return torch.cat(ba), torch.cat(bf)



def calculate_auc_pr(y_scores, y_true):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return [pr_auc]


