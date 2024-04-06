import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def split_time_series(train_valid, year_windwos = 6, stat = False):
    start_train = train_valid.index.min()
    end_train = start_train + pd.DateOffset(weeks=year_windwos * 52)
    max_date = train_valid.index.max() + pd.DateOffset(weeks = 4)

    train_set = []
    val_set = []
    tra_gt_set = []
    val_gt_set = []

    while True:
        if end_train + pd.DateOffset(weeks = 52) > max_date + pd.DateOffset(weeks = 4):
            break

        # 定义训练集和验证集的结束点
        start_val = end_train + pd.DateOffset(weeks = 1)
        end_val = start_val + pd.DateOffset(weeks=52)
        #start_train = start_train + pd.DateOffset(weeks=52)

        # 切分数据
        train = train_valid.loc[start_train:end_train]
        val = train_valid.loc[start_val:end_val]
        if stat is False:
            tra_gt = train.iloc[:,[0]]
            val_gt = val.iloc[:,[0]]

        train_set.append(train)
        val_set.append(val)
        if stat is False:
            tra_gt_set.append(tra_gt)
            val_gt_set.append(val_gt)

        # 将训练集的结束点向前滑动
        end_train = end_val
    return train_set,val_set,tra_gt_set,val_gt_set

def csv_reader(filename,predict_index, removed_factors):
    df = pd.read_csv(filename)
    virus_index = ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus','h1_pos','sh3_pos','b_pos','RSV_org']
    for virus in virus_index:
        if virus != predict_index and virus in df.columns:
            df[virus] = df[virus].shift(1)

    if removed_factors:
        df = df.drop(columns=removed_factors)

    for time_fea in ['monthid','yearid','weekid']:
        if time_fea in df.columns:
            if time_fea == 'monthid':
                df['month_sin'] = np.sin(2 * np.pi * df['monthid'].astype(int)/ 12)
                df['month_cos'] = np.cos(2 * np.pi * df['monthid'].astype(int) / 12)
            if time_fea == 'yearid':
                min_year = df['yearid'].astype(int).min()
                df['year_continuous'] = df['yearid'].astype(int) - min_year
            df = df.drop(columns = [time_fea])

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    insert_predict_col = df.pop(predict_index)
    df.insert(0,predict_index, insert_predict_col)
    return df


def arima_seq_mo(predict_index,removed_factors = None):
    if predict_index == "pos_index":
        data = pd.read_csv("data/pos_index_prediction.csv")
        #data = csv_reader("data/pos_index_prediction.csv",predict_index,removed_factors)
    elif "RSV" in predict_index:
        data = pd.read_csv("data/rsv_predction.csv")
        #data = csv_reader("data/rsv_predction.csv",predict_index,removed_factors)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

    data = data[predict_index]
    min_date = data.index.min()
    split_date = min_date + pd.DateOffset(weeks=52 * 12)
    train_valid = data[data.index <= split_date]
    test = data[data.index > split_date]



    train,val,tra_gt,val_gt = split_time_series(train_valid,stat = True)

    #train_valid = train_valid.diff().dropna()
    #print(diff_train)
    return train,val,test,train_valid