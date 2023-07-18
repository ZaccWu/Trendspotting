import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt

T = 110
data = {}
for t in range(T):
    loader=dict(np.load("data/synthetic/edge_t"+str(t)+".npz"))
    adj=scipy.sparse.csr_matrix((loader['adj_data'],loader['adj_indices'],loader['adj_indptr']),shape=loader['adj_shape'])
    adj = adj.todense()
    day_sales = np.sum(adj, axis=1).T.tolist()[0]
    data['t'+str(t)] = day_sales

data = pd.DataFrame(data)
data.to_csv('sales.csv')