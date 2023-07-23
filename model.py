import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

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
        self.att_lstm = ATT_LSTM(lag, in_dim, hid_dim, out_channels)
        self.gatconv_1 = GCNConv(in_channels, hid_dim, add_self_loops=True)
        self.gatconv_2 = GCNConv(hid_dim, out_channels, add_self_loops=True)
        self.linear_1 = nn.Linear(out_channels+fea_dim, out_dim)
        self.act_1 = nn.ReLU()

    def forward(self, data):
        # node_x: (batch*J, fea_dim), node_yx: (batch*J, K), edge_index: (2, batch*E), edge_weight: (E)
        node_x, node_yx, edge_index, edge_weight = data.x[0], data.x[1], data.edge_index, data.edge_attr
        x1 = self.att_lstm(node_yx.unsqueeze(-1))  # (batch*J, K) -> (batch*J, 1, hidden)
        x1 = torch.squeeze(x1)  # x1: (batch*J, hidden)
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gatconv_1(x2, edge_index, edge_weight)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gatconv_2(x2, edge_index, edge_weight) # x2: (batch*J, hidden)
        xcom1 = torch.cat([x2, node_x], dim=1) # xcom1: (batch*J, hidden+fea_dim)
        pred = self.act_1(self.linear_1(xcom1))  # (batch*J, hidden) -> (batch*J, out_channels)
        return pred.squeeze(-1)