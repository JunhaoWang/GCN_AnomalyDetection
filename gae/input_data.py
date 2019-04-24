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

def generate_dense_block_attribute(attr_mat, k, lam):
    # Search over optimal ways to inject dense blocks
    attr = attr_mat
    n_users = attr.shape[0]
    n_features = attr.shape[1]

    num = np.exp(lam * np.sum(attr, axis=0))
    num = np.asarray([min(i, 10000) for i in num])
    prob_distri = num / (np.sum(num))
    attr_idx = np.arange(n_features)
    for i in range(n_users):
        np.random.seed(i)
        indices = np.random.choice(attr_idx,
                                   size=k,
                                   p=prob_distri)
        attr[i][indices] = 1

    return attr

def format_data(adj_mat_path, attr_mat_path, use_features, adj_density):

    adj_mat = np.load(adj_mat_path)
    adj_mat = adj_mat.astype(np.uint8)
    attr_mat = np.load(attr_mat_path)

    mixed = np.concatenate((adj_mat, attr_mat), 1)
    input_size = mixed.shape[0]

    # First we inject dense blocks into the adjacency matrix
    blocks = [
        {'idx_range': (0, 1000), 'ajacency_density': float(adj_density)},
        {'idx_range': (2000, 3000), 'ajacency_density': float(adj_density)},
        {'idx_range': (5000, 6000), 'ajacency_density': float(adj_density)},
    ]

    for ind, b in enumerate(blocks):
        s_idx = b['idx_range'][0]
        e_idx = b['idx_range'][1]
        adj_mat[s_idx:e_idx, s_idx:e_idx] = generate_dense_block_ajacency(e_idx - s_idx,
                                                                          b['ajacency_density'])

    # Now we inject dense blocks into the attribute matrix
    # attr_mat = generate_dense_block_attribute(attr_mat, 3, 1e-9)
    attr_mat = generate_dense_block_attribute(attr_mat, 3, adj_density)

    # Setting labels to be 0 or 1 based on whether or not they correspond to an anomaly
    labels = np.zeros((input_size, 1))
    for block in blocks:
        idx_s, idx_e = block['idx_range']
        labels[idx_s:idx_e] = 1
    labels = labels.astype(np.uint8)

    # Convert to sparse matrices for fast processing
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
