import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import random
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser('Trendspotting')
# # task parameter
# parser.add_argument('--tau', type=int, help='tau-day-ahead prediction', default=1)
# # data parameter
parser.add_argument('--K', type=int, help='look-back window size', default=30)
parser.add_argument('--gth_pos', type=float, help='correlation threshold', default=0.0)
parser.add_argument('--exp_th', type=float, help='explore threshold', default=1.2)
# # loss parameter
# parser.add_argument('--reg1', type=float, help='reg1', default=1)
# parser.add_argument('--reg2', type=float, help='reg2', default=1)
# # training parameter
parser.add_argument('--seed', type=int, help='random seed', default=101)
parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--bs', type=int, help='batch size', default=32)
parser.add_argument('--n_epoch', type=int, help='number of epochs', default=200)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')


class time_att(nn.Module):
    def __init__(self, lag, n_hidden_1):
        super(time_att, self).__init__()
        self.W = nn.Parameter(torch.zeros(lag, n_hidden_1))
        nn.init.xavier_normal_(self.W.data)

    def forward(self, ht):
        ht_W = ht.mul(self.W)
        ht_W = torch.sum(ht_W, dim=2)
        att = F.softmax(ht_W, dim=1)
        return att

class ATT_LSTM(nn.Module):
    def __init__(self, lag, in_dim, n_hidden_1=32, out_dim=32):
        super(ATT_LSTM, self).__init__()
        self.LSTM = nn.LSTM(in_dim, n_hidden_1, 1, batch_first=True, bidirectional=False)
        self.time_att = time_att(lag, n_hidden_1)
        self.fc = nn.Sequential(nn.Linear(n_hidden_1, out_dim), nn.ReLU(True))

    def forward(self, x):
        ht, (hn, cn) = self.LSTM(x)
        t_att = self.time_att(ht).unsqueeze(dim=1)
        att_ht = torch.bmm(t_att, ht)
        att_ht = self.fc(att_ht)
        return att_ht

class TRENDSPOT2(torch.nn.Module):
    def __init__(self, lag, hid_dim=32, in_channels=32, out_channels=32, out_dim=1):
        super(TRENDSPOT2, self).__init__()
        self.dropout = 0.5
        self.training = True
        self.att_lstm_series = ATT_LSTM(lag, 3, hid_dim, out_channels)
        self.att_lstm_nodes = ATT_LSTM(lag, 1, hid_dim, out_channels)
        self.conv_1 = GCNConv(out_channels, hid_dim)
        self.conv_2 = GCNConv(hid_dim, out_channels)

        self.linear_sales = nn.Linear(out_channels*2, 1)
        self.act = nn.ReLU()

    def forward(self, graph):
        # input: graph batch
        # node_x: (fea_dim, K)*bs, edge_index: (E,2)*bs, edge_weight: (E)*bs
        node_x, edge_index, edge_weight = graph.x, graph.edge_index, graph.edge_attr

        series_fea = node_x.reshape(-1,3,30).transpose(2,1) # (bs*fea_dim, K)->(bs, fea_dim, K)->(bs, K, fea_dim)
        emb_series = self.att_lstm_series(series_fea)  # (bs, K, fea_dim) -> (bs, 1, hidden)
        x1s = torch.squeeze(emb_series)  # x1s: (bs, hidden)
        if len(x1s.shape)==1:
            x1s = x1s.unsqueeze(0)


        node_fea = node_x.unsqueeze(1).transpose(2,1) # (bs*fea_num, K)->(bs*fea_num, 1, K)->(bs*fea_num, K, 1)
        emb_node = self.att_lstm_nodes(node_fea)  # (bs*fea_num, K, 1) -> (bs*fea_num, 1, hidden)
        emb_node = torch.squeeze(emb_node) # (bs*fea_num, 1, hidden)

        x2n = self.conv_1(emb_node, edge_index, edge_weight)
        x2n = F.dropout(x2n, p=0.5, training=self.training)
        x2n = self.conv_2(x2n, edge_index, edge_weight) # x2n: (fea_num, hidden)
        x2n = global_mean_pool(x2n, graph.batch) # graph-level readout -> (bs, hidden)
        xcom2 = torch.cat([x1s, x2n], dim=1)    # xcom2: (bs, 2*hidden)
        pred = self.act(self.linear_sales(xcom2))
        return pred.squeeze(-1)



class TSDataset(Dataset):
    def __init__(self, list):
        super().__init__()
        self.data_graph = list
        target = []
        for i in self.data_graph:
            target.append(i.y)
        print("Explore product prec: ", np.mean(target))

    def len(self):
        return len(self.data_graph)

    def get(self, idx):
        data = self.data_graph[idx]
        return data

# def MSE(y_true, y_pred):
#     return np.mean((np.square(y_true - y_pred)))
#
# def MAE(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))


def contrastive_loss(target, pred_score, m=5):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 10e-10)
    #print(dev_score.device, pred_score.device,target.device, Rs.device, delta.device)
    cont_score = torch.max(torch.zeros(pred_score.shape).to(device), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def transfer_pred(out, threshold):
    pred = out.clone()
    pred[torch.where(out < threshold)] = 0
    pred[torch.where(out >= threshold)] = 1
    return pred

def plot_tSNE(Zu, pred_u, title):
    # dimension reduction
    ts = manifold.TSNE(n_components=2, init='pca', random_state=42)
    x_ts = ts.fit_transform(Zu) # x_ts: (N_user, 2)
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_final = (x_ts - x_min) / (x_max - x_min)
    S_data = np.hstack((x_final, np.array(pred_u).reshape(-1,1)))  # concat dim-reduced feature
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]}) # S_data: (N_sample, 3)
    colors = ['blue','red']

    plt.figure(figsize=(10, 10))
    for cla in range(2):
        X = S_data.loc[S_data['label'] == cla]['x']
        Y = S_data.loc[S_data['label'] == cla]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker='.', c=colors[cla], edgecolors=colors[cla], alpha=0.9,)
        plt.xticks([])
        plt.yticks([])
    plt.title(title, fontsize=32, fontweight='normal', pad=20)
    plt.savefig('img/embedding/'+title + '.jpg')
    #plt.show()

def construct_feature(series_fea_j, day):
    # series_fea_j: (time_step, fea_dim)
    y_past7 = series_fea_j[day-7:day,-1]
    y_head7 = series_fea_j[day:day+7,-1]
    if np.sum(y_past7)==0:
        y = 0
    else:
        y = np.sum(y_head7)/np.sum(y_past7)
    y_inc = 0 if y<=args.exp_th else 1
    series_fea = series_fea_j[day-args.K:day,:].T.astype(np.float64) # -> (fea_dim, K)

    wA = 1-pairwise_distances(series_fea,metric='correlation') # [-1,1] means correlation
    wA = np.array(wA)
    wA = np.nan_to_num(wA, copy=False)
    wA_pos = wA.copy()
    wA[wA < args.gth_pos] = 0
    wA_pos[wA_pos < args.gth_pos] = 0
    wA_pos[wA_pos >= args.gth_pos] = 1

    A_pos = sp.coo_matrix(wA_pos - np.eye(len(wA_pos)))
    pos_edge_index = np.array([A_pos.row, A_pos.col]).T # (num_edges, 2)

    edge_weight_index = []
    for e in pos_edge_index:
        i, j = e[0], e[1]
        edge_weight_index.append(wA[i,j])

    graph = Data(edge_index=torch.LongTensor(pos_edge_index).t().contiguous(),
                 edge_attr=torch.FloatTensor(edge_weight_index), x=torch.FloatTensor(series_fea), y=torch.tensor(y_inc), num_nodes=series_fea.shape[0])
    return graph


def main():
    # construct training & test samples
    train_sample_list, test_sample_list = [], []
    for j in range(100):
        y = series[j,:,:]
        series_fea_j = y[np.argsort(y[:,1])][:,2:]  # (time_step, fea_dim)
        for day in range(args.K, 70):
            graph = construct_feature(series_fea_j, day)
            train_sample_list.append(graph)
        for day in range(70,90):
            graph = construct_feature(series_fea_j, day)
            test_sample_list.append(graph)

    model = TRENDSPOT2(lag=args.K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_data, test_data = TSDataset(train_sample_list), TSDataset(test_sample_list)
    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False)

    result = {'epoch':[],'MSE':[],'MAE':[],'C1R1':[],'C1R2':[],'C1R3':[],'AUC':[],}
    for epoch in range(args.n_epoch):
        t0 = time.time()
        model.train()
        model.training = True
        total_loss = 0
        for graph in train_loader:
            optimizer.zero_grad()
            out = model(graph)
            y_label = graph.y
            class_loss = contrastive_loss(y_label, out)
            loss = class_loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        t1 = time.time()
        print("training loss:", total_loss)
        print("time of train epoch:", t1 - t0)

        model.eval()
        model.training = False

        true_inc_list = []
        pred_inc_list = []
        r1_list, r2_list, r3_list = [], [], []
        for tgraph in test_loader:
            true_inc = tgraph.y.detach().cpu().numpy()
            true_inc_list.extend(true_inc)
            pred_inc= model(tgraph)
            r1 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.5, dim=None, keepdim=False,
                                                        interpolation='higher')).detach().cpu().numpy()
            r2 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.7, dim=None, keepdim=False,
                                                        interpolation='higher')).detach().cpu().numpy()
            r3 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.8, dim=None, keepdim=False,
                                                        interpolation='higher')).detach().cpu().numpy()

            pred_inc_list.extend(pred_inc.detach().cpu().numpy())
            r1_list.extend(r1)
            r2_list.extend(r2)
            r3_list.extend(r3)


        r1_rec = classification_report(np.array(true_inc_list), np.array(r1_list), target_names=['class0', 'class1'],
                                        output_dict=True)['class1']['recall']
        r2_rec = classification_report(np.array(true_inc_list), np.array(r2_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']
        r3_rec = classification_report(np.array(true_inc_list), np.array(r3_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']
        auc = roc_auc_score(np.array(true_inc_list),np.array(pred_inc_list))


        result['epoch'].append(epoch)
        result['C1R1'].append(r1_rec)
        result['C1R2'].append(r2_rec)
        result['C1R3'].append(r3_rec)
        result['AUC'].append(auc)

        print("time of val epoch:", time.time() - t1)
        print('Epoch {:3d},'.format(epoch + 1),
              'C1R1 {:3f},'.format(r1_rec),
              'C1R2 {:3f},'.format(r2_rec),
              'C1R3 {:3f},'.format(r3_rec),
              'AUC {:3f},'.format(auc),
              'time {:3f}'.format(time.time() - t0))

    #result = pd.DataFrame(result)
    #result.to_csv('result_predLoss_aug'+str(args.reg1)+'_dec'+str(args.reg2)+'.csv', index=False)

if __name__ == '__main__':
    # dv
    series = np.load('data/dv_count.npy', allow_pickle=True)
    main()
