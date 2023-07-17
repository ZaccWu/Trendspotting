import numpy as np
import pandas as pd
import scipy.stats as st
import random
import math
import torch
import os

'''
default:
alpha=0.5
delta=0.02
'''
#alpha = 0.3
delta = 0.05

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

num_user = 10000
num_category = 50
num_product = 1000
T = 110

def trend(t, mu=3, sigma=0.5):
    return np.exp(-(np.log(t) - mu)**2 / (2 * sigma**2)) / (t * sigma * np.sqrt(2 * np.pi))


# data basic feature
zi = np.random.rand(num_user,2)    # (I,2)
zj = np.random.rand(num_product,2) # (J,2)
zc = np.random.rand(num_category,2) # (C,2)
gamma_c = np.random.rand(num_category)
gamma_j = np.random.rand(num_product)

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
    k = np.random.randint(1, 222)
    Vt.append(Vtype[k:k+T])
St, Vt = np.array(St), np.array(Vt)

for t in range(T):
    epsilon_ic = np.random.normal(0,0.2,(num_user, num_category))
    epsilon_ij = np.random.normal(0,0.2,(num_user, num_product))

    # u_ic/p_ic: (I,C), u_ij/p_ij: (I,J)
    u_ic = gamma_c[np.newaxis,:] + np.dot(zi,zc.T) + epsilon_ic
    u_ij = gamma_j[np.newaxis,:] + np.dot(zi,zj.T) + St[:,t][np.newaxis,:] + epsilon_ij
    p_ic = 1/(1+np.exp(-u_ic))
    p_ij = 1/(1+np.exp(-u_ij))
    print("Mean prob:", np.mean(p_ic), np.mean(p_ij))

    purchase_prob = []  # (J, I)
    for j in range(num_product):
        purchase_prob.append(p_ic[:,int(j/50)]*p_ij[:,j])

    purchase = np.random.binomial(1, np.array(purchase_prob))
    print("Total purchase perc.:", np.mean(purchase))






# pop_v = np.random.power(a=alpha, size=sample_content)
# pop_tag = pop_v.copy() # majority & minority
# pop_tag[pop_tag<np.mean(pop_v)] = 0
# pop_tag[pop_tag>=np.mean(pop_v)] = 1

# match_score = []
# for zu in Zu:
#     match_score.append(np.abs(Zv-zu))
# match_score=np.array(match_score)
#
#
# # data 1
# # generate click record
# edge_list = []
# for u in range(sample_user):
#     for v in range(sample_content):
#         # logit = pop_v[v]*match_score[u][v]/np.mean(pop_v) # normalize to make the density similar
#         logit = match_score[u][v]
#         click_prob = 1/(1+math.exp(-logit))*delta
#         click = np.random.binomial(1, click_prob)
#         if click==1:
#             edge_list.append([u,v])
# edge_list = np.array(edge_list) # (num_click, 2)
# print(edge_list.shape)
#
#
# # train-val-test
# num_click = len(edge_list)
# choice_idx = [i for i in range(num_click)]
# train_idx = np.random.choice(choice_idx, int(num_click*0.7), replace=False)
# val_idx = np.random.choice(choice_idx, int(num_click*0.15), replace=False)
# test_idx = np.random.choice(choice_idx, int(num_click*0.15), replace=False)
# edge_train, edge_val, edge_test = edge_list[train_idx], edge_list[val_idx], edge_list[test_idx]
# u_degree, v_degree = degree(torch.LongTensor(edge_train[:, 0])), degree(torch.LongTensor(edge_train[:, 1]))
#
# pop_tag = v_degree.numpy().copy() # majority & minority
# pop_tag[pop_tag<np.mean(pop_tag)] = 0
# pop_tag[pop_tag>=np.mean(pop_tag)] = 1
#
# # check the data
# print(edge_train.shape, edge_val.shape, edge_test.shape)
# print(pd.Series(u_degree).value_counts().sort_index(), pd.Series(v_degree).value_counts().sort_index())
# print("average and median degree of content:", np.mean(v_degree.numpy()), np.median(v_degree.numpy()))
#
# savedt_path = 'data/synthetic/d'+str(delta)+'/'
# if not os.path.isdir(savedt_path):
#     os.makedirs(savedt_path)
# # save data
# np.save(savedt_path+'train_edge.npy', edge_train)
# np.save(savedt_path+'val_edge.npy', edge_val)
# np.save(savedt_path+'test_edge.npy', edge_test)
# np.save(savedt_path+'Zu.npy', Zu)
# np.save(savedt_path+'Zv.npy', Zv)
# np.save(savedt_path+'poptag.npy', pop_tag)