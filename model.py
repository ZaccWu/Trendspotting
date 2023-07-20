import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

class TRENDSPOT(torch.nn.Module):
    def __init__(self, in_dim=8, hid_dim=64, in_channels=64, out_channels=32, heads=1, num_classes=3):
        super(TRENDSPOT, self).__init__()
        self.dropout = 0.5
        self.training = True
        self.att_lstm = LSTMHA(in_dim, hid_dim, 2, in_channels)
        self.gatconv_1 = GATConv(in_channels, out_channels, heads, dropout=self.dropout)
        self.gatconv_3 = GATConv(out_channels * heads, out_channels, 1, dropout=self.dropout)
        self.linear = nn.Linear(out_channels, num_classes)
        self.act = nn.ReLU()

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index # x0: (batch*num_stock, K, hidden)
        x1 = self.att_lstm(x0)  # x1: (batch*num_stock, 1, hidden)
        x1 = torch.squeeze(x1)  # x1: (batch*num_stock, hidden)
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.gatconv_1(x2, edge_index)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gatconv_3(x2, edge_index)
        x3 = self.act(self.linear(x2))  # x3: (batch*num_stock, hidden)
        #x3 = self.linear(x2)  # x3: (batch*num_stock, hidden)
        return F.log_softmax(x3, dim=1), x3