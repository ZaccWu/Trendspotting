import sys
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import random
import time
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import manifold

from model import TRENDSPOT


parser = argparse.ArgumentParser('Trendspotting')
# task parameter
parser.add_argument('--tau', type=int, help='tau-day-ahead prediction', default=1)
# data parameter
parser.add_argument('--K', type=int, help='look-back window size', default=30)
parser.add_argument('--exp_th', type=float, help='explore threshold', default=1.2)
# loss parameter
parser.add_argument('--reg1', type=float, help='reg1', default=1)
parser.add_argument('--reg2', type=float, help='reg2', default=1)
# training parameter
parser.add_argument('--seed', type=int, help='random seed', default=101)
parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--bs', type=int, help='batch size', default=16)
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



def MSE(y_true, y_pred):
    return np.mean((np.square(y_true - y_pred)))

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def data_transform(Matrix):
    Min, Max = np.min(Matrix), np.max(Matrix)
    result = (Matrix-Min)/(Max-Min)
    return result

def load_graph_data(day):
    # y_fea: （J, time_step, 1) ; y_pred: list (J)
    y_sale, y_fea = sales['t'+str(day)], np.array(sales.loc[:,'t'+str(day-args.K-7):'t'+str(day-1-7)])
    y_past7 = np.array(sales.loc[:,'t'+str(day-14):'t'+str(day-7)])
    y_head7 = np.array(sales.loc[:,'t'+str(day-7):'t'+str(day)])
    y_inc = np.sum(y_head7,axis=1)/np.sum(y_past7,axis=1)
    y_inc[y_inc <= args.exp_th] = 0
    y_inc[y_inc > args.exp_th] = 1
    print("Day: ", day, "Explore product prec: ", np.mean(y_inc))

    # features: (J, zdim)
    features = np.load("data/synthetic/product_x.npy")
    edge_val = np.load("data/synthetic/weight_t"+str(day)+".npy")
    edge_val = data_transform(edge_val)
    edges = Data(x=[torch.FloatTensor(features),torch.FloatTensor(y_fea)], edge_index=torch.LongTensor(edge_index).t().contiguous(),
                 y=[torch.FloatTensor(y_sale),torch.LongTensor(y_inc)], edge_attr=torch.FloatTensor(edge_val))
    return edges

def decorrelate(embI, embV):
    embI, embV = F.normalize(embI, dim=1), F.normalize(embV, dim=1)
    orthogonal = torch.sum(torch.mul(embI, embV), dim=1)
    return torch.sum(orthogonal)

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

def main():
    train_graph_list = []
    for day in range(37,90):
        graph_data = load_graph_data(day)
        train_graph_list.append([graph_data])

    test_graph_list = []
    for day in range(90,110):
        graph_data = load_graph_data(day)
        test_graph_list.append([graph_data])

    model = TRENDSPOT(lag=args.K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_graph_list, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_graph_list, shuffle=False)
    result = {'epoch':[],'MSE':[],'MAE':[],'C1R1':[],'C1R2':[],'C1R3':[],'AUC':[],}
    for epoch in range(args.n_epoch):
        t0 = time.time()
        model.train()
        model.training = True
        total_loss = 0

        for data in train_loader:
            data_fea = data[0].to(device)
            true_sales, true_inc = data_fea.y[0], data_fea.y[1]
            optimizer.zero_grad()
            out, out_star, out_inc, embI, embV = model(data_fea)
            main_loss = F.mse_loss(out, true_sales)
            aug_loss = F.mse_loss(out_star, true_sales)
            dec_loss = decorrelate(embI, embV)
            class_loss = contrastive_loss(true_inc, out_inc)
            #class_loss =criterion(out_inc, true_inc)
            #class_loss = ce_loss(out_inc, true_inc)

            loss = main_loss + class_loss * 6000 + aug_loss * args.reg1 + dec_loss * args.reg2
            #print(main_loss.item(), class_loss.item(), aug_loss.item(), dec_loss.item())
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        t1 = time.time()
        print("training loss:", total_loss)
        print("time of train epoch:", t1 - t0)

        model.eval()
        model.training = False
        true_sales_list = []
        pred_sales_list = []
        true_inc_list = []
        pred_inc_list = []
        r1_list, r2_list, r3_list = [], [], []

        label_table = []
        pred_table = []
        embI, embV = [], []


        for tdata in test_loader:
            tdata_fea = tdata[0]
            true_sales, true_inc = tdata_fea.y[0], tdata_fea.y[1]
            true_sales = true_sales.detach().cpu().numpy()
            true_inc = true_inc.detach().cpu().numpy()

            true_sales_list.extend(true_sales)
            true_inc_list.extend(true_inc)

            tdata_fea = tdata_fea.to(device)
            pred_sales, _, pred_inc, zI, zV = model(tdata_fea)

            embI.extend(zI.detach().cpu().numpy())
            embV.extend(zV.detach().cpu().numpy())

            pred_sales = pred_sales.detach().cpu().numpy()
            pred_sales_list.extend(pred_sales)
            label_table.append(true_sales)
            pred_table.append(pred_sales)


            r1 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.8, dim=None, keepdim=False,
                                                        interpolation='higher')).detach().cpu().numpy()
            r2 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.9, dim=None, keepdim=False,
                                                        interpolation='higher')).detach().cpu().numpy()
            r3 = transfer_pred(pred_inc, torch.quantile(pred_inc, 0.95, dim=None, keepdim=False,
                                                        interpolation='higher')).detach().cpu().numpy()

            pred_inc_list.extend(pred_inc.detach().cpu().numpy())
            r1_list.extend(r1)
            r2_list.extend(r2)
            r3_list.extend(r3)

        if epoch%5==0:
            plot_tSNE(np.array(embI), np.array(true_inc_list), 'epoch_' + str(epoch) + '_embI')
            plot_tSNE(np.array(embV), np.array(true_inc_list), 'epoch_' + str(epoch) + '_embV')


        # for visualing the prediction
        # if epoch == args.n_epoch - 1:
        #     J = len(label_table[0])
        #     label_table = pd.DataFrame(label_table)
        #     label_table.columns = ['p'+str(j) for j in range(J)]
        #     pred_table = pd.DataFrame(pred_table)
        #     pred_table.columns = ['p'+str(j)+'_pred' for j in range(J)]
        #     pred_label_table = pd.concat([label_table,pred_table],axis=1)
        #     pred_label_table.to_csv('result_predVal.csv', index=False)

        mse = MSE(np.array(true_sales_list), np.array(pred_sales_list))
        mae = MAE(np.array(true_sales_list), np.array(pred_sales_list))
        # c1_recall = classification_report(np.array(true_inc_list), np.array(pred_inc_list), target_names=['class0', 'class1'],
        #                                output_dict=True)['class1']['recall']
        # f1 = classification_report(np.array(true_inc_list), np.array(pred_inc_list), target_names=['class0', 'class1'],
        #                                output_dict=True)['macro avg']['f1-score']

        r1_rec = classification_report(np.array(true_inc_list), np.array(r1_list), target_names=['class0', 'class1'],
                                        output_dict=True)['class1']['recall']
        r2_rec = classification_report(np.array(true_inc_list), np.array(r2_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']
        r3_rec = classification_report(np.array(true_inc_list), np.array(r3_list), target_names=['class0', 'class1'],
                                       output_dict=True)['class1']['recall']
        auc = roc_auc_score(np.array(true_inc_list),np.array(pred_inc_list))


        result['epoch'].append(epoch)
        result['MSE'].append(mse)
        result['MAE'].append(mae)
        result['C1R1'].append(r1_rec)
        result['C1R2'].append(r2_rec)
        result['C1R3'].append(r3_rec)
        result['AUC'].append(auc)

        print("time of val epoch:", time.time() - t1)
        print('Epoch {:3d},'.format(epoch + 1),
              'MSE {:3f},'.format(mse),
              'MAE {:3f},'.format(mae),
              'C1R1 {:3f},'.format(r1_rec),
              'C1R2 {:3f},'.format(r2_rec),
              'C1R3 {:3f},'.format(r3_rec),
              'AUC {:3f},'.format(auc),
              'time {:3f}'.format(time.time() - t0))

    result = pd.DataFrame(result)
    result.to_csv('result_predLoss_aug'+str(args.reg1)+'_dec'+str(args.reg2)+'.csv', index=False)

if __name__ == '__main__':
    # dv
    sales = pd.read_csv('sales.csv')
    edge_index = np.load("data/synthetic/edge_weight_idx.npy")
    main()
