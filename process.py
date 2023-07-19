import numpy as np
import pandas as pd
import scipy.sparse
import torch

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

# dv
sales = pd.read_csv('sales.csv')
y = sales['t0']

# features: (J,zdim)
features = np.load("data/synthetic/product_x.npy")
loader=dict(np.load("data/synthetic/edge_t0.npz"))
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

edges = Data(x=torch.tensor(features), edge_index=torch.LongTensor(edge_index).t().contiguous(), y=torch.tensor(y), edge_attr=torch.tensor(edge_val))