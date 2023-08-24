import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

seed = 101
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    def __init__(self, lag, in_dim=1, n_hidden_1=32, out_dim=32):
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

class TRENDSPOT(torch.nn.Module):
    def __init__(self, lag, in_dim=1, fea_dim=4, hid_dim=32, in_channels=32, out_channels=32, out_dim=1):
        super(TRENDSPOT, self).__init__()
        self.dropout = 0.5
        self.training = True
        self.att_lstm_I = ATT_LSTM(lag, in_dim, hid_dim, out_channels)
        self.att_lstm_V = ATT_LSTM(lag, in_dim, hid_dim, out_channels)
        self.gatconv_I1 = GCNConv(in_channels+fea_dim, hid_dim, add_self_loops=True)
        self.gatconv_I2 = GCNConv(hid_dim, out_channels, add_self_loops=True)
        self.gatconv_V1 = GCNConv(in_channels+fea_dim, hid_dim, add_self_loops=True)
        self.gatconv_V2 = GCNConv(hid_dim, out_channels, add_self_loops=True)

        # self.linear_sales = nn.Sequential(
        #     nn.Linear(out_channels*2, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, 1)
        # )
        # self.linear_inc = nn.Sequential(
        #     nn.Linear(out_channels*2, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, 1)
        # )

        self.linear_sales = nn.Linear(out_channels*2, 1)
        self.linear_inc_V = nn.Linear(out_channels, 1)
        self.linear_inc_I = nn.Linear(out_channels, 1)
        self.act = nn.ReLU()

    def forward(self, data):
        # node_x: (batch*J, fea_dim), node_yx: (batch*J, K), edge_index: (2, batch*E), edge_weight: (E)
        node_x, node_yx, edge_index, edge_weight = data.x[0], data.x[1], data.edge_index, data.edge_attr
        x1I = self.att_lstm_I(node_yx.unsqueeze(-1))  # (batch*J, K) -> (batch*J, 1, hidden)
        x1V = self.att_lstm_V(node_yx.unsqueeze(-1))  # (batch*J, K) -> (batch*J, 1, hidden)
        x1I, x1V = torch.squeeze(x1I), torch.squeeze(x1V)  # x1: (batch*J, hidden)
        xcom1I = torch.cat([x1I, node_x], dim=1)  # xcom1: (batch*J, hidden+fea_dim)
        xcom1V = torch.cat([x1V, node_x], dim=1)  # xcom1: (batch*J, hidden+fea_dim)

        x2I = F.dropout(xcom1I, self.dropout, training=self.training)
        x2I = self.gatconv_I1(x2I, edge_index, edge_weight)
        x2I = F.dropout(x2I, self.dropout, training=self.training)
        x2I = self.gatconv_I2(x2I, edge_index, edge_weight) # x2I: (batch*J, hidden)
        x2V = F.dropout(xcom1V, self.dropout, training=self.training)
        x2V = self.gatconv_V1(x2V, edge_index, edge_weight)
        x2V = F.dropout(x2V, self.dropout, training=self.training)
        x2V = self.gatconv_V2(x2V, edge_index, edge_weight) # x2V: (batch*J, out_channels)
        x2V_star = x2V[torch.randperm(x2V.size(0))]

        xcom2 = torch.cat([x2I, x2V], dim=1)    # xcom2: (batch*J, out_channels*2)
        xcom2_star = torch.cat([x2I, x2V_star], dim=1)  # xcom2_star: (batch*J, out_channels*2)

        # (batch*J, out_channels*2) -> (batch*J, 1)
        pred = self.act(self.linear_sales(xcom2))
        pred_Vstar = self.act(self.linear_sales(xcom2_star))
        # (batch*J, out_channels) -> (batch*J, 1)
        pred_inc = self.act(self.linear_inc_V(x2V)+self.linear_inc_I(x2I))

        return pred.squeeze(-1), pred_Vstar.squeeze(-1), pred_inc, x2I, x2V