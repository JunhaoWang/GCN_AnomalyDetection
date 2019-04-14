import scipy.sparse as sp
import numpy as np
import scipy.io
import inspect
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple


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

def format_data(adj_mat_path, attr_mat_path, use_features):

    adj_mat = np.load(adj_mat_path)
    attr_mat = np.load(attr_mat_path)

    adj_mat_length = adj_mat.shape[0]
    indices = np.arange(adj_mat_length)
    np.random.RandomState(seed=42).permutation(indices)[:int(0.02 * adj_mat_length)]

    labels = np.zeros((adj_mat_length,))
    labels[indices] = 1
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
