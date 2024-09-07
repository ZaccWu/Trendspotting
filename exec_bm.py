import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import manifold

from model_base import LSTM_X, GRU_X

parser = argparse.ArgumentParser('Trendspotting')
# # task parameter
parser.add_argument('--data_file', type=str, help='path of the data set', default='data/datasample2.csv')
parser.add_argument('--result_path', type=str, help='path of the result file', default='result/ts_benchmark/')
parser.add_argument('--gen_dt', type=bool, help='whether construct sample', default=False)  # fixed: False (=True only when constructing new data)
parser.add_argument('--model', type=str, help='choose model', default='lstm') # 'lstm', 'gru'
parser.add_argument('--train_loss', type=str, help='choose model', default='contrast') # 'contrast', 'evl'

# # data parameter
parser.add_argument('--K', type=int, help='look-back window size', default=30)
parser.add_argument('--pts', type=int, help='time for evaluate explosive', default=3)   # fix this parameter!
parser.add_argument('--gth_pos', type=float, help='correlation threshold', default=0.2) # used in gen_dt process
parser.add_argument('--exp_th', type=float, help='explore threshold', default=2.21) # large data: 2.21, used in gen_dt process
# threshold in sample_dv (3 days): 3.33(top1%), 2.54(top2%), 2.21(top3%), 1.91(top5%), 1.56(top10%), 1.29(top20%)

# # training parameter
parser.add_argument('--regr', type=float, help='reconstruct reg', default=0.01) # 0.01
parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--bs', type=int, help='batch size', default=8192) # default 8192, as large as possible
parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100) # 100


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
EPS = 1e-15

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TSADataset(Dataset):
    def __init__(self, x_l, y_l):
        super(TSADataset, self).__init__()
        self.x = torch.FloatTensor(x_l)
        self.y = torch.FloatTensor(y_l)
        print("Explore product prec: ", torch.mean(self.y))
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return len(self.y)

def contrastive_loss(target, pred_score, m=5):
    # target: 0-1, pred_score: float
    Rs = torch.mean(pred_score)
    delta = torch.std(pred_score)
    dev_score = (pred_score - Rs)/(delta + 1e-10)
    #print(dev_score.device, pred_score.device,target.device, Rs.device, delta.device)
    cont_score = torch.max(torch.zeros(pred_score.shape).to(device), m-dev_score)
    loss = dev_score[(1-target).nonzero()].sum()+cont_score[target.nonzero()].sum()
    return loss

def ev_loss(target, pred_score):  # 235032
    # gamma=1.0 version
    prop_0 = len((1-target).nonzero())  # label = 0
    prop_1 = len(target.nonzero())      # label = 1
    pred_score_sigmoid = torch.sigmoid(pred_score)
    pos_loss = -torch.log(pred_score_sigmoid[target.nonzero()] + EPS).mean() * (prop_0/(prop_0+prop_1))
    neg_loss = -torch.log(1 - pred_score_sigmoid[(1-target).nonzero()] + EPS).mean() * (prop_1/(prop_0+prop_1))
    loss = pos_loss + neg_loss
    return loss

def feature_loss(pred_fea, true_fea):
    # input: (bs, fea_dim, K)
    assert len(pred_fea.shape)>2
    assert len(true_fea.shape)>2
    scale = true_fea.shape[1] * true_fea.shape[2]
    criterion = nn.MSELoss()
    pred_fea_flat = pred_fea.reshape(pred_fea.size(0), -1)
    true_fea_flat = true_fea.reshape(true_fea.size(0), -1)
    return criterion(pred_fea_flat, true_fea_flat)/scale

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
    if np.sum(y_past)==0:
        y = 0
    else:
        y = np.sum(y_head)/np.sum(y_past)
    y_inc = 0 if y<=args.exp_th else 1
    series_fea = series_fea_j[day-args.K:day,:].T # -> (fea_dim, K)
    return series_fea, y_inc

def construct_train_test_sample():
    series = np.load('data/dv_count2.npy', allow_pickle=True)
    print("Construct Dt [1] Input shape: ", series.shape) # (num_stock, time_step, fea_dim)
    num_content = series.shape[0]
    # construct train/val/test samples
    train_x_list, val_x_list, test_x_list, train_y_list, val_y_list, test_y_list = [], [], [], [], [], []
    for j in range(num_content):
        content_fea = series[j,:,:]
        series_fea_j = content_fea[np.argsort(content_fea[:,1])][:,2:]  # (time_step, fea_dim), remove 'content_id', 'visite_time'
        for day in range(args.K, 61):
            train_x, train_y = construct_feature(series_fea_j, day)
            train_x_list.append(train_x)
            train_y_list.append(train_y)
        for day in range(61, 71):
            val_x, val_y = construct_feature(series_fea_j, day)
            val_x_list.append(val_x)
            val_y_list.append(val_y)
        for day in range(71, 92-args.pts):
            test_x, test_y = construct_feature(series_fea_j, day)
            test_x_list.append(test_x)
            test_y_list.append(test_y)

    train_x_list, train_y_list = np.array(train_x_list), np.array(train_y_list)
    val_x_list, val_y_list = np.array(val_x_list), np.array(val_y_list)
    test_x_list, test_y_list = np.array(test_x_list), np.array(test_y_list)
    print("Construct Dt [2] Final shape")
    print("Train: ", train_x_list.shape, train_y_list.shape)
    print("Val: ", val_x_list.shape, val_y_list.shape)
    print("Test: ", test_x_list.shape, test_y_list.shape)

    train_data, val_data, test_data = TSADataset(train_x_list, train_y_list), TSADataset(val_x_list, val_y_list), TSADataset(test_x_list, test_y_list)

    if not os.path.isdir('task_data/'):
        os.makedirs('task_data/')

    with open('task_data/train_dt.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('task_data/val_dt.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open('task_data/test_dt.pkl', 'wb') as f:
        pickle.dump(test_data, f)


def main():
    if args.gen_dt == True:
        construct_train_test_sample()

    fea_dim = 10
    print("[1] Start loading...")
    with open('task_data/train_dt.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('task_data/val_dt.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open('task_data/test_dt.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print("[2] Finish loading...")

    train_loader = DataLoader(train_data, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.bs, shuffle=False)
    print(len(train_loader), len(val_loader), len(test_loader))

    if args.model == 'lstm':
        model = LSTM_X(lag=args.K, fea_dim=fea_dim, encoderh_dim=64, decoderh_dim=32).to(device)
    elif args.model == 'gru':
        model = GRU_X(lag=args.K, fea_dim=fea_dim, encoderh_dim=64, decoderh_dim=32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_val_metrics = -np.inf
    result = {'epoch':[],'r1_rec':[],'r2_rec':[],'r3_rec':[],'r1_ndcg':[],'r2_ndcg':[],'r3_ndcg':[],'AUC':[]}
    for epoch in range(args.n_epoch):
        model.train()
        model.training = True
        total_loss = 0
        for feature, label in train_loader:
            feature, label = feature.to(device), label.to(device)

            optimizer.zero_grad()
            pred_y, tsa_emb = model(feature)
            pred_tsa = model.decoder(tsa_emb)

            if args.train_loss == 'evl':
                class_loss = ev_loss(label, pred_y)
            else:
                class_loss = contrastive_loss(label, pred_y)

            reconst_loss = feature_loss(pred_tsa, feature)
            loss = class_loss + reconst_loss * args.regr
            # print("Pred/Rec loss: ", class_loss.item(), reconst_loss.item())
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        model.eval()
        model.training = False
        true_inc_list, pred_inc_list = [], []
        r1_list, r2_list, r3_list = [], [], []
        for vfeature, vlabel in val_loader:
            vfeature, vlabel = vfeature.to(device), vlabel.to(device)
            true_inc = vlabel.detach().cpu().numpy()
            true_inc_list.extend(true_inc)
            pred_inc, _ = model(vfeature)
            r2 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.9, dim=None, keepdim=False))
            pred_inc_list.extend(pred_inc.detach().cpu().numpy())
            r2_list.extend(r2.detach().cpu().numpy())
        r2_rec_val = classification_report(np.array(true_inc_list), np.array(r2_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']

        true_inc_list, pred_inc_list = [], []
        r1_list, r2_list, r3_list = [], [], []
        for tfeature, tlabel in test_loader:
            tfeature, tlabel = tfeature.to(device), tlabel.to(device)
            true_inc = tlabel.detach().cpu().numpy()
            true_inc_list.extend(true_inc)
            pred_inc, _ = model(tfeature)
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

        print('Epoch {:3d},'.format(epoch + 1),
              'r1_rec {:3f},'.format(r1_rec),
              'r2_rec {:3f},'.format(r2_rec),
              'r3_rec {:3f},'.format(r3_rec),
              'r1_ndcg {:3f},'.format(r1_ndcg),
              'r2_ndcg {:3f},'.format(r2_ndcg),
              'r3_ndcg {:3f},'.format(r3_ndcg),
              'AUC {:3f},'.format(auc))

        result['epoch'].append(epoch)
        result['r1_rec'].append(r1_rec)
        result['r2_rec'].append(r2_rec)
        result['r3_rec'].append(r3_rec)
        result['r1_ndcg'].append(r1_ndcg)
        result['r2_ndcg'].append(r2_ndcg)
        result['r3_ndcg'].append(r3_ndcg)
        result['AUC'].append(auc)

        # Model selection
        if r2_rec_val > max_val_metrics:
            max_val_metrics = r2_rec_val
            Rep_ts_rec1, Rep_ts_rec2, Rep_ts_rec3, Rep_ts_auc = r1_rec, r2_rec, r3_rec, auc
            Rep_ts_ndcg1, Rep_ts_ndcg2, Rep_ts_ndcg3 = r1_ndcg, r2_ndcg, r3_ndcg

    if not os.path.isdir(args.result_path):
        os.makedirs(args.result_path)
    result = pd.DataFrame(result)
    result.to_csv(args.result_path+args.model+'_seed'+str(seed)+'.csv', index=False)

    return Rep_ts_rec1, Rep_ts_rec2, Rep_ts_rec3, Rep_ts_ndcg1, Rep_ts_ndcg2, Rep_ts_ndcg3, Rep_ts_auc


if __name__ == '__main__':
    Rep_res = {'seed': [], 'rec1': [], 'rec2': [], 'rec3': [], 'ndcg1': [], 'ndcg2': [], 'ndcg3': [], 'auc': []}  # all possible evaluation metrics
    for seed in range(1, 2):
        set_seed(seed)

        Rep_ts_rec1, Rep_ts_rec2, Rep_ts_rec3, Rep_ts_ndcg1, Rep_ts_ndcg2, Rep_ts_ndcg3, Rep_ts_auc = main()
        Rep_res['seed'].append(seed)
        Rep_res['rec1'].append(Rep_ts_rec1)
        Rep_res['rec2'].append(Rep_ts_rec2)
        Rep_res['rec3'].append(Rep_ts_rec3)
        Rep_res['ndcg1'].append(Rep_ts_ndcg1)
        Rep_res['ndcg2'].append(Rep_ts_ndcg2)
        Rep_res['ndcg3'].append(Rep_ts_ndcg3)
        Rep_res['auc'].append(Rep_ts_auc)

        print(seed, ' Rec1 {:.4f}, '.format(Rep_ts_rec1), 'Rec2 {:.4f}, '.format(Rep_ts_rec2),
              'Rec3 {:.4f}, '.format(Rep_ts_rec3), 'Ndcg1 {:.4f}, '.format(Rep_ts_ndcg1),
              'Ndcg2 {:.4f}, '.format(Rep_ts_ndcg2), 'Ndcg3 {:.4f}, '.format(Rep_ts_ndcg3),
              'AUC {:.4f}, '.format(Rep_ts_auc))

    print(' Rec1 {:.4f} ({:.4f}), '.format(np.mean(Rep_res['rec1']), np.std(Rep_res['rec1'])),
          ' Rec2 {:.4f} ({:.4f}), '.format(np.mean(Rep_res['rec2']), np.std(Rep_res['rec2'])),
          ' Rec3 {:.4f} ({:.4f}), '.format(np.mean(Rep_res['rec3']), np.std(Rep_res['rec3'])),
          ' Ndcg1 {:.4f} ({:.4f}), '.format(np.mean(Rep_res['ndcg1']), np.std(Rep_res['ndcg1'])),
          ' Ndcg2 {:.4f} ({:.4f}), '.format(np.mean(Rep_res['rec2']), np.std(Rep_res['ndcg2'])),
          ' Ndcg3 {:.4f} ({:.4f}), '.format(np.mean(Rep_res['ndcg3']), np.std(Rep_res['ndcg3'])),
          ' AUC {:.4f} ({:.4f}), '.format(np.mean(Rep_res['auc']), np.std(Rep_res['auc'])))
    print('-----------------------------------------------')


