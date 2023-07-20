import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import random
import time
import argparse

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from model import TRENDSPOT

parser = argparse.ArgumentParser('Trendspotting')
# task parameter
parser.add_argument('--tau', type=int, help='tau-day-ahead prediction', default=1)
# data parameter
parser.add_argument('--K', type=int, help='look-back window size', default=30)
# training parameter
parser.add_argument('--seed', type=int, help='random seed', default=101)
parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--bs', type=int, help='batch size', default=16)
parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100)


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


# dv
sales = pd.read_csv('sales.csv')

def load_graph_data(day):
    # y_fea: （J, time_step, 1) ; y_pred: list (J)
    y_pred, y_fea = sales['t'+str(day)], torch.tensor(np.array(sales.loc[:,'t'+str(day-args.K):'t'+str(day-1)]))
    # features: (J, zdim)
    features = np.load("data/synthetic/product_x.npy")
    edge_val = np.load("data/synthetic/weight_t"+str(day)+".npy")
    edges = Data(x=[torch.tensor(features),y_fea], edge_index=torch.LongTensor(edge_index).t().contiguous(), y=torch.tensor(y_pred), edge_attr=torch.tensor(edge_val))
    return edges

edge_index = np.load("data/synthetic/edge_weight_idx.npy")
train_graph_list = []
for day in range(30,90):
    graph_data = load_graph_data(day)
    train_graph_list.append([graph_data])
test_graph_list = []
for day in range(90,110):
    graph_data = load_graph_data(day)
    train_graph_list.append([graph_data])

model = TRENDSPOT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_loader = DataLoader(train_graph_list, batch_size=args.bs, shuffle=True)
test_loader = DataLoader(test_graph_list, batch_size=args.bs)

for epoch in range(args.n_epoch):
    t0 = time.time()
    model.train()
    model.training = True
    total_loss = 0
    for data in train_loader:
        data_fea = data[0].to(device)
        optimizer.zero_grad()
        out = model(data_fea)
        loss = None # TODO: define the loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    t1 = time.time()
    print("training loss:", total_loss)
    print("time of train epoch:", t1 - t0)

    model.eval()
    model.training = False
    train_label_list = []
    train_pred_list = []
    for tdata in train_loader:
        tdata_fea = tdata[0]
        train_label_list.extend(tdata_fea.y.detach().cpu().numpy())
        tdata_fea = tdata_fea.to(device)
        tr_pred, z = model(tdata_fea)
        tr_pred = reclassificaiton(tr_pred)
        _, tr_pred = tr_pred.max(dim=1)
        train_pred_list.extend(tr_pred.detach().cpu().numpy())
    train_acc, train_mac_f1 = ld.metrics(train_pred_list, train_label_list)

    label_list = []
    pred_list = []
    for tdata in test_loader:
        tdata_fea = tdata[0]
        label_list.extend(tdata_fea.y.detach().cpu().numpy())
        tdata_fea = tdata_fea.to(device)
        pred, z = model(tdata_fea)
        pred = reclassificaiton(pred)
        _, pred = pred.max(dim=1)
        pred_list.extend(pred.detach().cpu().numpy())
    acc, mac_f1 = ld.metrics(pred_list, label_list)

    Epoch.append(epoch + 1)
    Tr_acc.append(train_acc)
    Tr_mf1.append(train_mac_f1)
    Ts_acc.append(acc)
    Ts_mf1.append(mac_f1)
    print("time of val epoch:", time.time() - t1)
    print('Epoch {:3d},'.format(epoch + 1),
          'Train Accuracy {:.4f}'.format(train_acc),
          'Train Macro_f1 {:.4f}'.format(train_mac_f1),
          'Accuracy {:.4f},'.format(acc),
          'Macro_f1 {:.4f},'.format(mac_f1),
          'time {:4f}'.format(time.time() - t0))
    print(classification_report(np.array(label_list), np.array(pred_list)))
    print('task:', args.mpath, 'regp:', args.regp, 'regn:', args.regn, ' tau:', args.tau)