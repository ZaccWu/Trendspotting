
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_dim, hd_dim, out_dim):
        super(Decoder, self).__init__()
        self.LSTM = nn.LSTM(in_dim, hd_dim, 1, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(nn.Linear(hd_dim, out_dim), nn.LeakyReLU())
    def forward(self, emb):
        ht, (hn, cn) = self.LSTM(emb) # ht: (bs, K, hidden)
        pred_fea = self.fc(ht) # pred_fea: (bs, K, out_dim)
        return pred_fea

# reusable module
class ATT_LSTM(nn.Module):
    def __init__(self, lag, in_dim, n_hidden_1=32, out_dim=32):
        super(ATT_LSTM, self).__init__()
        self.LSTM = nn.LSTM(in_dim, n_hidden_1, 1, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(nn.Linear(n_hidden_1, out_dim), nn.ReLU(True))
        self.W = nn.Parameter(torch.zeros(lag, n_hidden_1))
        nn.init.xavier_normal_(self.W.data)
    def forward(self, x):
        ht, (hn, cn) = self.LSTM(x)
        ht_W = torch.sum(ht.mul(self.W), dim=2)
        t_att = F.softmax(ht_W, dim=1).unsqueeze(dim=1)
        att_ht = torch.bmm(t_att, ht)
        att_ht = self.fc(att_ht)
        return att_ht

# possible architectures
class Lstm_Attention(nn.Module):
    def __init__(self, lag, fea_dim, hid_dim=32, out_channels=32, out_dim=1):
        super(Lstm_Attention, self).__init__()
        self.training = True
        self.fea_dim = fea_dim
        self.z_dim = out_channels
        self.att_lstm_series = ATT_LSTM(lag, self.fea_dim, hid_dim, out_channels)
        self.linear_trend = nn.Linear(out_channels, 1)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        series_x = x.transpose(2,1) # (bs, fea_dim, K)->(bs, K, fea_dim)
        tsa_emb = self.att_lstm_series(series_x).squeeze(dim=1)
        pred_trend = self.act(self.linear_trend(tsa_emb))
        return pred_trend.squeeze(-1), tsa_emb

class Lstm(nn.Module):
    def __init__(self, lag, fea_dim, hid_dim=32, out_channels=32, out_dim=1):
        super(Lstm, self).__init__()
        self.LSTM = nn.LSTM(fea_dim, hid_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.linear_trend = nn.Linear(out_channels, 1)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        series_x = x.transpose(2,1) # (bs, fea_dim, K)->(bs, K, fea_dim)
        tsa_emb, (hn, cn) = self.LSTM(series_x) # -> (bs, K, hidden)
        tsa_emb = tsa_emb[:, -1, :]  # -> (bs, hidden)
        pred_trend = self.act(self.linear_trend(tsa_emb))
        return pred_trend.squeeze(-1), tsa_emb

class Gru(nn.Module):
    def __init__(self, lag, fea_dim, hid_dim=32, out_channels=32, out_dim=1):
        super(Gru, self).__init__()
        self.GRU = nn.GRU(fea_dim, hid_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.linear_trend = nn.Linear(out_channels, 1)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        series_x = x.transpose(2,1) # (bs, fea_dim, K)->(bs, K, fea_dim)
        tsa_emb, _ = self.GRU(series_x) # -> (bs, K, hidden)
        tsa_emb = tsa_emb[:,-1,:] # -> (bs, hidden)
        pred_trend = self.act(self.linear_trend(tsa_emb))
        return pred_trend.squeeze(-1), tsa_emb
