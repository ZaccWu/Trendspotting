import os
import sys
import numpy as np
import collections
import pandas as pd
import scipy.sparse as sp
import torch
import random
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Function
from typing import Any, Optional, Tuple

# from torch_geometric.data import Data, Dataset
# from torch_geometric.loader import DataLoader

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import manifold
from sklearn.metrics.pairwise import pairwise_distances

parser = argparse.ArgumentParser('Trendspotting')
# # task parameter
# parser.add_argument('--tau', type=int, help='tau-day-ahead prediction', default=1)
parser.add_argument('--data_file', type=str, help='path of the data set', default='data/datasample2.csv')
parser.add_argument('--result_path', type=str, help='path of the result file', default='result/ts_240114/')
parser.add_argument('--gen_dt', type=bool, help='whether construct sample', default=False)
parser.add_argument('--online', type=bool, help='whether use online data', default=True)   # alibaba: True

parser.add_argument('--model', type=str, help='choose model', default='full') # 'full', 'wo_ts', 'wo_g'

# # data parameter
parser.add_argument('--K', type=int, help='look-back window size', default=30)
parser.add_argument('--pts', type=int, help='time for evaluate explosive', default=3)   # fix this parameter!
parser.add_argument('--gth_pos', type=float, help='correlation threshold', default=0.2) # used in gen_dt process
parser.add_argument('--exp_th', type=float, help='explore threshold', default=2.21) # large data: 2.21, used in gen_dt process
# threshold in sample_dv (3 days): 3.33(top1%), 2.54(top2%), 2.21(top3%), 1.91(top5%), 1.56(top10%), 1.29(top20%)

# loss parameter
parser.add_argument('--reg1', type=float, help='reg1', default=1e-4)  # 1e-4
parser.add_argument('--reg2', type=float, help='reg2', default=0.1)     # 0.1
parser.add_argument('--reg3', type=float, help='reg3', default=1e-5)  # 1e-4
parser.add_argument('--reg4', type=float, help='reg4', default=1)       # 1
# # training parameter
parser.add_argument('--seed', type=int, help='random seed', default=101)
parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--bs', type=int, help='batch size', default=1024)           # as large as possible
parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100) # 100


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

# get the data
def get_odps_data(table_txt):
    with open(table_txt, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def read_data_to_dict(datafile, data_dict):
    data = get_odps_data(datafile)
    head = data.split('\n')[0].split('||$||')
    data = data.split('||$||')
    item_len = len(head) - 1
    for i in range(1, int(len(data) // item_len) - 1):
        rowdata = data[i * item_len:(i + 1) * item_len + 1]
        rowdata[0] = rowdata[0].split('\n')[1]
        rowdata[-1] = rowdata[-1].split('\n')[0]
        item = collections.OrderedDict(list(zip(head, rowdata)))
        data_dict[i] = item

        try:
            a = int(item['content_id'])
        except Exception as e:
            print(e)
            print(i, item)
        if 'content_id' not in item:
            print(i, item)
        if i % 10000 == 0:
            print('finish {}'.format(i))
            print(data_dict[i])
    return data_dict


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(torch.nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()
    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

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

class MTView(nn.Module):
    def __init__(self, lag, fea_dim, hid_dim=32, out_channels=32):
        super(MTView, self).__init__()
        # affine transformation
        self.fea_dim = fea_dim
        self.spv_aff = nn.Linear(self.fea_dim, self.fea_dim)
        self.tpv_aff = nn.Linear(self.fea_dim, self.fea_dim)

        self.att_lstm_series = ATT_LSTM(lag, self.fea_dim, hid_dim, out_channels*2)
        self.att_lstm1 = ATT_LSTM(lag, self.fea_dim, hid_dim, out_channels)
        self.att_lstm2 = ATT_LSTM(lag, self.fea_dim, hid_dim, out_channels)
        self.att_lstm3 = ATT_LSTM(lag, self.fea_dim, hid_dim, out_channels)
        self.att_lstm4 = ATT_LSTM(lag, self.fea_dim, hid_dim, out_channels)
        self.view_cat = nn.Linear(out_channels*3, out_channels*2)
        self.classifinet = nn.Sequential(nn.Linear(out_channels, 2))
        self.grl = GRL_Layer()

    def forward(self, x):
        # input (x): (bs, fea_dim, K)
        series_x = x.transpose(2,1) # (bs, fea_dim, K)->(bs, K, fea_dim)
        series_spv = self.spv_aff(F.normalize(series_x, dim=2)) # (bs, K, fea_dim) -> (bs, K, fea_dim')
        series_tpv = self.tpv_aff(F.normalize(series_x, dim=1)) # (bs, K, fea_dim) -> (bs, K, fea_dim')

        tsa_emb = self.att_lstm_series(series_x).squeeze(dim=1)

        spv_specific = self.att_lstm1(series_spv).squeeze(dim=1)   # (bs, K, fea_dim) -> (bs, 1, hidden) -> (bs, hidden)
        spv_shared = self.att_lstm2(series_spv).squeeze(dim=1)
        tpv_specific = self.att_lstm3(series_tpv).squeeze(dim=1)
        tpv_shared = self.att_lstm4(series_tpv).squeeze(dim=1)

        discri_shared = torch.cat([spv_shared, tpv_shared], dim=0) # (bs+bs, hidden) for GRL classification
        # (bs, hidden) * (hidden, bs) -> (bs, bs) -> 1
        orthogonal = torch.abs(torch.mm(spv_specific, spv_shared.T).sum()) \
                     + torch.abs(torch.mm(tpv_specific, tpv_shared.T).sum())

        allv_shared = (spv_shared + tpv_shared)/2   # (bs, hidden)
        tsa_emb_norm = self.view_cat(torch.cat([spv_specific, tpv_specific, allv_shared], dim=-1)) # (bs, hidden*3) -> (bs, hidden*2)
        view_prob = self.classifinet(self.grl(discri_shared)) # (bs+bs, hidden) -> (bs+bs, 2)
        return torch.cat([tsa_emb, tsa_emb_norm], dim=-1), orthogonal, view_prob



class TRENDSPOT2(torch.nn.Module):
    def __init__(self, lag, fea_dim, hid_dim=32, out_channels=32, out_dim=1):
        super(TRENDSPOT2, self).__init__()
        self.training = True
        self.fea_dim = fea_dim
        self.z_dim = out_channels

        self.mt_view1 = MTView(lag, fea_dim, hid_dim, out_channels)
        self.mt_view2 = MTView(lag, fea_dim, hid_dim, out_channels)
        self.mt_view_shared = MTView(lag, fea_dim, hid_dim, out_channels)

        self.linear_sales1 = nn.Linear(out_channels * 8, 1)
        self.linear_sales2 = nn.Linear(out_channels * 8, 1)
        self.linear_sales3 = nn.Linear(out_channels * 8, 1)
        self.linear_salesn = nn.Linear(out_channels * 8, 1)

        self.linear_trend = nn.Linear(out_channels*8 + 4, 1)
        #self.linear_trend = nn.Linear(out_channels * 2, 1)
        self.act = nn.LeakyReLU()



    def forward(self, x):
        zI_emb, orth_I, view_prob_I = self.mt_view1(x)
        zV_emb, orth_V, view_prob_V = self.mt_view2(x)
        zS_emb, orth_S, view_prob_S = self.mt_view2(x)

        zI = torch.cat([zI_emb, zS_emb], dim=1)
        zV = torch.cat([zV_emb, zS_emb], dim=1)
        #zV_star = zV[torch.randperm(zV.size(0))]
        x1s_star = torch.cat([zI], dim=1)
        pred_sales1 = self.act(self.linear_sales1(x1s_star))  # (bs, hidden) -> (bs,1)
        pred_sales2 = self.act(self.linear_sales1(x1s_star))
        pred_sales3 = self.act(self.linear_sales1(x1s_star))
        pred_salesn = self.act(self.linear_sales1(x1s_star))

        pred_trend = self.act(self.linear_trend(torch.cat([zV,pred_sales1,pred_sales2,pred_sales3,pred_salesn], dim=-1)))

        view_prob = torch.cat([view_prob_I, view_prob_V, view_prob_S], dim=0)
        orthogonal = orth_I + orth_V + orth_S
        return [pred_sales1.squeeze(-1), pred_sales2.squeeze(-1), pred_sales3.squeeze(-1), pred_salesn.squeeze(-1)],\
               pred_trend.squeeze(-1), zI_emb, zV_emb, orthogonal, view_prob.squeeze(-1)




class TSADataset(Dataset):
    def __init__(self, x_l, y_l, y_ls):
        super(TSADataset, self).__init__()
        self.x = torch.FloatTensor(x_l)
        self.y = torch.FloatTensor(y_l)
        self.ys = torch.FloatTensor(y_ls)
        print("Explore product prec: ", torch.mean(self.y))
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.ys[idx]
    def __len__(self):
        return (len(self.y))

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

def find_positions_in_vru(list_a, list_b):
    positions_dict = {}  # 创建一个字典用于存储列B表中每个元素的位置
    for i, element in enumerate(list_b):
        positions_dict[element] = i
    result = []  # 用于存储列表A中每个元素在列表B中的位置
    for element in list_a:
        if element in positions_dict:
            result.append(positions_dict[element])
    return result

def cal_ndcgK(vcu, vru):
    # vru是正序的排名(按照推荐指数)
    position = find_positions_in_vru(vcu, vru)
    dcg = np.sum([1/np.log2(2+i) for i in position])
    inter_len = len(set(vru) & set(vcu))
    idcg = np.sum([1/np.log2(2+i) for i in range(inter_len)])
    if idcg == 0:
        return 0
    return dcg/idcg

def decorrelate(embI, embV):
    # orthogonal = torch.abs(torch.mm(embI, embV.T).sum())
    # return orthogonal
	embI, embV = F.normalize(embI, dim=1), F.normalize(embV, dim=1)
	orthogonal = torch.abs(torch.sum(torch.mul(embI, embV), dim=1))
	return torch.sum(orthogonal)

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
    series_fea_j = series_fea_j.astype(np.float64)
    y_past = series_fea_j[day-args.pts:day,-1]
    y_head = series_fea_j[day:day+args.pts,-1]
    y_sales1 = series_fea_j[day + args.pts - 3, -1]
    y_sales2 = series_fea_j[day + args.pts - 2, -1]
    y_sales3 = series_fea_j[day + args.pts - 1, -1]
    y_salesn = (y_sales1+y_sales2+y_sales3)/3

    if np.sum(y_past)==0:
        y = 0
    else:
        y = np.sum(y_head)/np.sum(y_past)
    y_inc = 0 if y<=args.exp_th else 1
    series_fea = series_fea_j[day-args.K:day,:].T # -> (fea_dim, K)
    return series_fea, y_inc, [y_sales1, y_sales2, y_sales3, y_salesn]


def main():
    if args.online == True:
        # load data
        data_dict = read_data_to_dict(args.data_file, {})
        df = pd.DataFrame.from_dict(data_dict).T
        # record all the existing values
        date_ = sorted(df['visite_time'].unique())
        content_ = sorted(df['content_id'].unique())
        columns = df.columns
        # select variables in interest (y is the last)
        dv_col = ['content_id', 'visite_time', 'click_uv_1d', 'consume_uv_1d_valid', 'favor_uv_1d', 'comment_uv_1d',
                  'share_uv_1d', 'collect_uv_1d', 'attention_uv_1d', 'lead_shop_uv_1d', 'cart_uv_1d', 'consume_uv_1d']
        sample_content_dv = []
        start_time = time.time()
        st = 0
        for c in content_:
            dt = df.query('content_id == "' + c + '"')
            dt.sort_values('visite_time')
            content_c_feature = np.array(dt[dv_col])
            st+=1   # 有部分content是不够97天的
            if len(content_c_feature) == len(date_):
                sample_content_dv.append(content_c_feature)
        end_time = time.time()
        print("Test running time: ", end_time - start_time)
        series = np.array(sample_content_dv)
        fea_dim = len(dv_col)-2

    else:
        series = np.load('data/dv_count2.npy', allow_pickle=True)
        fea_dim = series.shape[-1]-2

    print("[0] input shape: ", series.shape)
    num_content = series.shape[0]
    # construct training & test samples
    if args.gen_dt == True:
        train_x_list, test_x_list, train_y_list, test_y_list = [], [], [], []
        train_ys_list, test_ys_list = [], []
        for j in range(num_content):
            y = series[j,:,:]
            series_fea_j = y[np.argsort(y[:,1])][:,2:]  # (time_step, fea_dim), remove 'content_id', 'visite_time'
            for day in range(args.K, 70):
                train_x, train_y, train_ys = construct_feature(series_fea_j, day)
                train_x_list.append(train_x)
                train_y_list.append(train_y)
                train_ys_list.append(train_ys)
            for day in range(70, 92-args.pts):
                test_x, test_y, test_ys = construct_feature(series_fea_j, day)
                test_x_list.append(test_x)
                test_y_list.append(test_y)
                test_ys_list.append(test_ys)

        train_x_list, train_y_list, test_x_list, test_y_list = np.array(train_x_list), np.array(train_y_list), np.array(test_x_list), np.array(test_y_list)
        train_ys_list, test_ys_list = np.array(train_ys_list), np.array(test_ys_list)
        train_data, test_data = TSADataset(train_x_list, train_y_list, train_ys_list), TSADataset(test_x_list, test_y_list, test_ys_list)

        if not os.path.isdir('task_data/'):
            os.makedirs('task_data/')
        # torch.save(train_data, 'task_data/train_dt.pt')
        # torch.save(test_data, 'task_data/test_dt.pt')
        with open('task_data/train_dt.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open('task_data/test_dt.pkl', 'wb') as f:
            pickle.dump(test_data, f)

    print("[1] Start loading...")
    # train_data = torch.load('task_data/train_dt.pt')
    # test_data = torch.load('task_data/test_dt.pt')
    # 加载数据集
    with open('task_data/train_dt.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('task_data/test_dt.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print("[2] Finish loading...")

    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False)

    if args.model == 'full':
        model = TRENDSPOT2(lag=args.K, fea_dim=fea_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    result = {'epoch':[],'r1_rec':[],'r2_rec':[],'r3_rec':[],'r1_ndcg':[],'r2_ndcg':[],'r3_ndcg':[],'AUC':[],'time':[]}
    for epoch in range(args.n_epoch):
        t0 = time.time()
        model.train()
        model.training = True
        total_loss = 0
        for feature, label, sales in train_loader:
            feature, label, sales = feature.to(device), label.to(device), sales.to(device)

            optimizer.zero_grad()
            out_sales, out_y, zI, zV, ortho_val, view_prob = model(feature)
            target_viewlabel = torch.cat([torch.ones(len(label)), torch.zeros(len(label)),
                                          torch.ones(len(label)), torch.zeros(len(label)),
                                          torch.ones(len(label)), torch.zeros(len(label))], dim=-1).to(device)

            class_loss = contrastive_loss(label, out_y)
            _, yidx = out_y.topk(len(label)-len(label.nonzero()), largest=False)

            sales_loss = F.mse_loss(out_sales[0][yidx], sales[:,0][yidx]) + F.mse_loss(out_sales[1][yidx], sales[:,1][yidx]) \
                         + F.mse_loss(out_sales[2][yidx], sales[:,2][yidx]) + F.mse_loss(out_sales[3][yidx], sales[:,3][yidx])

            dec_loss = decorrelate(zI, zV)
            orth_loss = ortho_val
            view_loss = torch.nn.CrossEntropyLoss()(view_prob, target_viewlabel.long())

            #print(class_loss.item(), sales_loss.item()*args.reg1, dec_loss.item()*args.reg2, orth_loss.item()* args.reg3, view_loss.item() * args.reg4)

            loss = class_loss + sales_loss*args.reg1 + dec_loss*args.reg2 + orth_loss * args.reg3 + view_loss * args.reg4
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        t1 = time.time()
        #print("training loss:", total_loss)
        #print("time of train epoch:", t1 - t0)

        model.eval()
        model.training = False

        true_inc_list = []
        pred_inc_list = []
        r1_list, r2_list, r3_list = [], [], []
        for tfeature, tlabel, tsales in test_loader:
            tfeature, tlabel, tsales = tfeature.to(device), tlabel.to(device), tsales.to(device)
            true_inc = tlabel.detach().cpu().numpy()
            true_inc_list.extend(true_inc)
            _, pred_inc, _, _, _, _ = model(tfeature)
            r1 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.95, dim=None, keepdim=False))
            r2 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.9, dim=None, keepdim=False))
            r3 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.8, dim=None, keepdim=False))

            pred_inc_list.extend(pred_inc.detach().cpu().numpy())
            r1_list.extend(r1.detach().cpu().numpy())
            r2_list.extend(r2.detach().cpu().numpy())
            r3_list.extend(r3.detach().cpu().numpy())


        r1_rec = classification_report(np.array(true_inc_list), np.array(r1_list), target_names=['class0', 'class1'],
                                        output_dict=True)['class1']['recall']
        r2_rec = classification_report(np.array(true_inc_list), np.array(r2_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']
        r3_rec = classification_report(np.array(true_inc_list), np.array(r3_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']
        auc = roc_auc_score(np.array(true_inc_list),np.array(pred_inc_list))

        _, pred_r1 = torch.topk(torch.tensor(pred_inc_list), k=len(np.nonzero(np.array(r1_list))[0]))  # pred_r1: (k_rec_content)
        _, pred_r2 = torch.topk(torch.tensor(pred_inc_list), k=len(np.nonzero(np.array(r2_list))[0]))
        _, pred_r3 = torch.topk(torch.tensor(pred_inc_list), k=len(np.nonzero(np.array(r3_list))[0]))

        r1_ndcg = cal_ndcgK(np.nonzero(np.array(true_inc_list))[0], pred_r1.numpy())
        r2_ndcg = cal_ndcgK(np.nonzero(np.array(true_inc_list))[0], pred_r2.numpy())
        r3_ndcg = cal_ndcgK(np.nonzero(np.array(true_inc_list))[0], pred_r3.numpy())

        result['epoch'].append(epoch)
        result['r1_rec'].append(r1_rec)
        result['r2_rec'].append(r2_rec)
        result['r3_rec'].append(r3_rec)
        result['r1_ndcg'].append(r1_ndcg)
        result['r2_ndcg'].append(r2_ndcg)
        result['r3_ndcg'].append(r3_ndcg)
        result['AUC'].append(auc)
        result['time'].append(time.time() - t0)

        #print("time of val epoch:", time.time() - t1)
        print('Epoch {:3d},'.format(epoch + 1),
              'r1_rec {:3f},'.format(r1_rec),
              'r2_rec {:3f},'.format(r2_rec),
              'r3_rec {:3f},'.format(r3_rec),
              'r1_ndcg {:3f},'.format(r1_ndcg),
              'r2_ndcg {:3f},'.format(r2_ndcg),
              'r3_ndcg {:3f},'.format(r3_ndcg),
              'AUC {:3f},'.format(auc),
              'time {:3f}'.format(time.time() - t0))

    result = pd.DataFrame(result)
    result.to_csv(args.result_path+args.model+'_result_rA'+str(args.reg1)+'_rB'+str(args.reg2)+'_rC'+str(args.reg3)+'_rD'+str(args.reg4)+'.csv', index=False)

if __name__ == '__main__':
    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)
    main()
