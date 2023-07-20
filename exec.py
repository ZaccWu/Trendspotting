import numpy as np
import pandas as pd
import scipy.sparse
import torch

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

tau = 1 # pred ahead
K = 30  # look back window size

# dv
sales = pd.read_csv('sales.csv')

def load_graph_data(day):
    # y_fea: （J, time_step, 1) ; y_pred: list (J)
    y_pred, y_fea = sales['t'+str(day)], torch.tensor(np.array(sales.loc[:,'t'+str(day-K):'t'+str(day-1)]))
    # features: (J, zdim)
    features = np.load("data/synthetic/product_x.npy")
    loader=dict(np.load("data/synthetic/edge_t"+str(day)+".npz"))
    # adj.shape: (J,I)
    adj=scipy.sparse.csr_matrix((loader['adj_data'],loader['adj_indices'],loader['adj_indptr']),shape=loader['adj_shape'])
    num_product, num_user = adj.shape
    neg_dict = {}
    for i in range(num_product):
        neg_dict[i] = set(adj[i].nonzero()[1])
    edge_index = []
    edge_val = []
    for i in range(num_product):
        for j in range(i+1,num_product):
            num_com_neg = len(neg_dict[i]&neg_dict[j])
            edge_index.append([i,j])
            edge_index.append([j,i])
            edge_val.append(num_com_neg)
            edge_val.append(num_com_neg)

    edges = Data(x=[torch.tensor(features),y_fea], edge_index=torch.LongTensor(edge_index).t().contiguous(), y=torch.tensor(y_pred), edge_attr=torch.tensor(edge_val))
    return edges

train_graph_list = []
for day in range(30,90):
    graph_data = load_graph_data(day)
    print(graph_data.x)
    train_graph_list.append([graph_data])
test_graph_list = []
for day in range(90,110):
    graph_data = load_graph_data(day)
    train_graph_list.append([graph_data])