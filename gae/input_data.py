import scipy.sparse as sp
import numpy as np
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple

import networkx as nx


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(data_source):
    data = scipy.io.loadmat("data/{}.mat".format(data_source))
    labels = data["Label"]
    attributes = sp.csr_matrix(data["Attributes"])
    network = sp.lil_matrix(data["Network"])

    return network, attributes, labels

def generate_dense_block_ajacency(node_size, density):
    edge_size = int(node_size * (node_size-1) / 2 * density)
    return nx.adjacency_matrix(
        nx.generators.dense_gnm_random_graph(node_size,edge_size), weight=None
    ).A

def format_data(adj_mat_path, attr_mat_path, use_features):

    adj_mat = np.load(adj_mat_path)
    adj_mat = adj_mat.astype(np.uint8)
    attr_mat = np.load(attr_mat_path)

    # adj_mat_length = adj_mat.shape[0]
    # indices = np.arange(adj_mat_length)
    # np.random.RandomState(seed=42).permutation(indices)[:int(0.02 * adj_mat_length)]
    # labels = np.zeros((adj_mat_length,))
    # labels[indices] = 1

    mixed = np.concatenate((adj_mat, attr_mat), 1)
    input_shape = mixed.shape[1]
    input_size = mixed.shape[0]
    tracking_artificial_anomaly = np.zeros(input_size)
    tracking_artificial_anomaly_aggregate = []

    blocks = [
        {'idx_range': (0, 1000), 'ajacency_density': .1, 'attribute_density': .3},
        {'idx_range': (2000, 3000), 'ajacency_density': .15, 'attribute_density': .3},
        {'idx_range': (5000, 6000), 'ajacency_density': .2, 'attribute_density': .3},
    ]

    for ind, b in enumerate(blocks):
        s_idx = b['idx_range'][0]
        e_idx = b['idx_range'][1]
        adj_mat[s_idx:e_idx, s_idx:e_idx] = generate_dense_block_ajacency(e_idx - s_idx, b['ajacency_density'])
        tracking_artificial_anomaly[s_idx:e_idx] = ind + 10
        for i in range(s_idx, e_idx):
            tracking_artificial_anomaly_aggregate.append(i)

    labels = np.zeros((input_size, 1))
    for block in blocks:
        idx_s, idx_e = block['idx_range']
        labels[idx_s:idx_e] = 1
    labels = labels.astype(np.uint8)

    adj = sp.lil_matrix(adj_mat)
    features = sp.csr_matrix(attr_mat)

    if use_features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, adj_norm, adj_label, features, labels]
    feas = {}
    for item in items:
        # item_name = [ k for k,v in locals().iteritems() if v == item][0]]
        item_name = retrieve_name(item)
        feas[item_name] = item

    return feas

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var and "item" not in var_name][0]
