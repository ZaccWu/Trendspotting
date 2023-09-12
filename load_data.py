import sys
import os
import random
import scipy.sparse as sp
import collections
import pandas as pd
import numpy as np
import torch
import argparse


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
        # print(df['content_type'].value_counts())
        # print(df['account_role'].value_counts())
        print(df['lead_pay_visitor_ids'].value_counts())
        print(df['visite_time'].value_counts())



def dataAnalysis():
    dataLoader = DataLoad()




if __name__ == '__main__':
    dataAnalysis()
