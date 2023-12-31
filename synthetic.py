import numpy as np
import pandas as pd
import scipy.sparse
import random
import math
import torch
import os


beta_m = 0.8
beta_v = 1
beta_s = 0.2
delta = 0.05

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

num_user = 3000
num_category = 20
num_product = 200
T = 110

def trend(t, mu=3, sigma=0.5):
    return 100*np.exp(-(np.log(t) - mu)**2 / (2 * sigma**2)) / (t * sigma * np.sqrt(2 * np.pi))


# data basic feature
zi = np.random.rand(num_user,2)    # (I,2)
zj = np.random.rand(num_product,2) # (J,2)
zc = np.random.rand(num_category,2) # (C,2)
gamma_c = np.random.rand(num_category)
gamma_j = np.random.rand(num_product)
product_feature = []
for j in range(num_product):
    j_feature = np.concatenate([zj[j,:],zc[int(j/50),:]])
    product_feature.append(j_feature)
product_feature = np.array(product_feature) # (J,4)

# purchase data
St, Vt = [], [] # (J,T)
t_idx = np.arange(T)
St1 = np.sin(t_idx*2*math.pi/52)
St2 = np.cos(t_idx*2*math.pi/52)
Vtype = np.concatenate([np.zeros(T),trend(np.arange(1,T+1)),np.zeros(T)])
for j in range(num_product):
    # seasonal trend
    if j%2 == 0:
        St.append(St2)
    else:
        St.append(St1)
    # fashion trend
    k = np.random.randint(0, 2*T-1)
    Vt.append(list(Vtype[k:k+T]))
St, Vt = np.array(St), np.array(Vt)

savedt_path = 'data/synthetic/'
if not os.path.isdir(savedt_path):
    os.makedirs(savedt_path)
np.save(savedt_path+'product_x.npy', product_feature)
edge_weight_index = []
for i in range(num_product):
    for j in range(i + 1, num_product):
        edge_weight_index.append([i, j])
        edge_weight_index.append([j, i])
edge_weight_index = np.array(edge_weight_index)
np.save(savedt_path+'edge_weight_idx.npy', edge_weight_index)

data = {}
for t in range(T):
    epsilon_ic = np.random.normal(0,0.2,(num_user, num_category))
    epsilon_ij = np.random.normal(0,0.2,(num_user, num_product))

    # u_ic/p_ic: (I,C), u_ij/p_ij: (I,J)
    u_ic = gamma_c[np.newaxis,:] + np.dot(zi,zc.T) + epsilon_ic
    u_ij = gamma_j[np.newaxis,:] + beta_m * np.dot(zi,zj.T) + beta_s * St[:,t][np.newaxis,:] + beta_v * Vt[:,t][np.newaxis,:] + epsilon_ij
    p_ic = 1/(1+np.exp(-u_ic))
    p_ij = 1/(1+np.exp(-u_ij))

    purchase_prob = []  # (J, I)
    for j in range(num_product):
        purchase_prob.append(p_ic[:,int(j/50)]*p_ij[:,j]*delta)

    purchase = np.random.binomial(1, np.array(purchase_prob))
    adj = scipy.sparse.csr_matrix(purchase)
    print("Day: ", t, "Purchase perc.:", np.mean(purchase), "shape:", purchase.shape)
    # calculate edge weights
    neg_dict = {}
    for i in range(num_product):
        neg_dict[i] = set(adj[i].nonzero()[1])
    edge_val = []
    for i in range(num_product):
        for j in range(i+1,num_product):
            num_com_neg = len(neg_dict[i]&neg_dict[j])
            edge_val.append(num_com_neg)
            edge_val.append(num_com_neg)
    edge_val = np.array(edge_val)
    np.save(savedt_path + 'weight_t'+str(t)+'.npy', edge_val)
    # calculate sales
    adj = adj.todense()
    day_sales = np.sum(adj, axis=1).T.tolist()[0]
    data['t'+str(t)] = day_sales

data = pd.DataFrame(data)
data.to_csv('sales.csv')

