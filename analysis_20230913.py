import sys
import os
import random
import scipy.sparse as sp
import collections
import pandas as pd
import numpy as np
import torch
import argparse
import time

parser = argparse.ArgumentParser('Trendspotting')

parser.add_argument('--file', type=str, help='path of the data set', default='data/datasample.csv')
parser.add_argument('--seed', type=int, help='random seed', default=101)
parser.add_argument('--gpu', type=int, help='gpu', default=0)

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


class DataLoad():
    def __init__(self):
        data_dict = read_data_to_dict(args.file, {})
        df = pd.DataFrame.from_dict(data_dict).T
        print(df['lead_pay_visitor_ids'].value_counts())

        # record all the existing values
        # date_ = sorted(df['visite_time'].unique())
        # content_ = sorted(df['content_id'].unique())
        # columns = df.columns
        # print("content_type:", df['content_type'].value_counts())
        # print("account_role:", df['account_role'].value_counts())
        # ana_col = columns[4:100]
        # df[ana_col] = df[ana_col].astype(float)
        # ana_dict = {
        #     'max': df[ana_col].max(),
        #     'mean': df[ana_col].mean(),
        #     'std': df[ana_col].std(),
        #     'q1': df[ana_col].quantile(q=0.1),
        #     'q2': df[ana_col].quantile(q=0.2),
        #     'q3': df[ana_col].quantile(q=0.3),
        #     'q4': df[ana_col].quantile(q=0.4),
        #     'q5': df[ana_col].quantile(q=0.5),
        #     'q6': df[ana_col].quantile(q=0.6),
        #     'q7': df[ana_col].quantile(q=0.7),
        #     'q8': df[ana_col].quantile(q=0.8),
        #     'q9': df[ana_col].quantile(q=0.9),
        # }
        # ana_dt = pd.DataFrame(ana_dict)
        #
        # dv_col = ['content_id','visite_time','click_uv_1d','consume_uv_1d','consume_uv_1d_valid']
        # sample_content_dv = []
        # start_time = time.time()
        # for c in content_[:40000]:
        #     dt = df.query('content_id == "' + c + '"')
        #     dt.sort_values('visite_time')
        #     content_c_feature = np.array(dt[dv_col])
        #     if len(content_c_feature) == len(date_):
        #         sample_content_dv.append(content_c_feature)
        # end_time = time.time()
        # print("Test running time: ", end_time-start_time)
        # sample_content_dv = np.array(sample_content_dv)
        #
        # if not os.path.isdir(result_path):
        #     os.makedirs(result_path)
        # ana_dt.to_csv(result_path+'analysis_count.csv')
        # np.save(result_path+'dv_count.npy', sample_content_dv)


        # # other analysis
        # feature_col = ['content_id','visite_time','expo_uv_1d','click_uv_1d','consume_uv_1d']
        # self.all_content_feature = []  # (content_num, time_step, feature_dim)
        # for c in content_:
        #     dt = df.query('content_id == "'+c+'"')
        #     dt.sort_values('visite_time')
        #     content_c_feature = np.array(dt[feature_col])   # (time_step, feature_dim)
        #     self.all_content_feature.append(content_c_feature)

        # content_record = {}
        # selected_cols = ['content_id','visite_time','expo_uv_1d','click_uv_1d','consume_uv_1d']
        # for i in range(len(df)):
        #     record = df.iloc[i]
        #     content_record.setdefault(record['content_id'], []).append(record[selected_cols])
        # for c in content_:
        #     content_record[c] = pd.DataFrame(content_record[c])
        #
        # print(content_record)

        #
        # for c in content_:
        #     u, v = true_click[click][0].item(), true_click[click][1].item()
        #     v_click_by_u_dict.setdefault(u, []).append(v)

        # for i in content_:
        #     print(df[df['content_id']==i][['content_id','visite_time','expo_pv_1d','expo_uv_1d']])

        #print(df['content_type'].value_counts())
        #print(df['account_role'].value_counts())
        # print(df['lead_pay_visitor_ids'].value_counts())
        #print(df['visite_time'].value_counts())



def dataAnalysis():
    dataLoader = DataLoad()

if __name__ == '__main__':
    result_path = 'result/ts_230913/'
    dataAnalysis()
    # a = np.load(result_path+'dv_count.npy',allow_pickle=True)
    # print(a)
