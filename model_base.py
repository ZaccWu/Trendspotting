
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_dim, hd_dim, out_dim):
        super(Decoder, self).__init__()
        self.LSTM = nn.LSTM(in_dim, hd_dim, 1, batch_first=True, bidirectional=False)
        self.dll = nn.Linear(hd_dim, out_dim)
        self.act = nn.LeakyReLU()
    def forward(self, emb):
        tsa_pred, (hn, cn) = self.LSTM(emb) # ht: (bs, K, hidden)
        tsa_pred = self.act(self.dll(tsa_pred)) # tsa_pred: (bs, K, out_dim)
        return tsa_pred

class LSTM_X(nn.Module):
    def __init__(self, lag, fea_dim, encoderh_dim, decoderh_dim):
        super(LSTM_X, self).__init__()
        self.LSTM = nn.LSTM(fea_dim, encoderh_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.linear_trend = nn.Linear(encoderh_dim, 1)
        self.act = nn.LeakyReLU()
        self.decoder = Decoder(in_dim=encoderh_dim, hd_dim=decoderh_dim, out_dim=fea_dim)
    def forward(self, x):
        series_x = x.transpose(2,1) # (bs, fea_dim, K)->(bs, K, fea_dim)
        tsa_emb, (_, _) = self.LSTM(series_x)
        ht = tsa_emb[:, -1, :]  # -> (bs, hidden)
        pred_trend = self.act(self.linear_trend(ht))
        return pred_trend.squeeze(-1), tsa_emb # tsa_emb: (bs, K, hidden)

class GRU_X(nn.Module):
    def __init__(self, lag, fea_dim, encoderh_dim, decoderh_dim):
        super(GRU_X, self).__init__()
        self.GRU = nn.GRU(fea_dim, encoderh_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.linear_trend = nn.Linear(encoderh_dim, 1)
        self.act = nn.LeakyReLU()
        self.decoder = Decoder(in_dim=encoderh_dim, hd_dim=decoderh_dim, out_dim=fea_dim)
    def forward(self, x):
        series_x = x.transpose(2,1) # (bs, fea_dim, K)->(bs, K, fea_dim)
        tsa_emb, (_, _) = self.GRU(series_x) # -> (bs, K, hidden)
        ht = tsa_emb[:,-1,:] # -> (bs, hidden)
        pred_trend = self.act(self.linear_trend(ht))
        return pred_trend.squeeze(-1), tsa_emb # tsa_emb: (bs, K, hidden)
