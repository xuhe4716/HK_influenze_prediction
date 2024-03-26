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

def split_time_series(train_valid, year_windwos = 6):
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
        tra_gt = train.iloc[:,[0]]
        val_gt = val.iloc[:,[0]]

        train_set.append(train)
        val_set.append(val)
        tra_gt_set.append(tra_gt)
        val_gt_set.append(val_gt)

        # 将训练集的结束点向前滑动
        end_train = end_val
    return train_set,val_set,tra_gt_set,val_gt_set

def csv_reader(filename,predict_index, removed_factors):
    df = pd.read_csv(filename)
    virus_index = ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus']
    for virus in virus_index:
        if virus != predict_index:
            df[virus] = df[virus].shift(1)

    if removed_factors:
        df = df.drop(columns=removed_factors)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    insert_predict_col = df.pop(predict_index)
    df.insert(0,predict_index, insert_predict_col)
    return df


def nn_seq_mo(seq_len, B, num,predict_index,removed_factors = None):
    print(predict_index)
    data = csv_reader("data/data_pp.csv",predict_index,removed_factors)

    min_date = data.index.min()
    split_date = min_date + pd.DateOffset(weeks=52 * 12)
    train_valid = data[data.index <= split_date]
    test = data[data.index > split_date]
    test_ground_truth = test.iloc[:,[0]]

    m, n = np.max(train_valid[train_valid.columns[0]]), np.min(train_valid[train_valid.columns[0]])


    # Min-Max Scaler
    columns = train_valid.columns

    scaler = MinMaxScaler()
    train_valid[columns] = scaler.fit_transform(train_valid[columns])
    test[columns] = scaler.transform(test[columns])

    train,val,tra_gt,val_gt = split_time_series(train_valid)

    def rolling_data(dataset, batch_size, shuffle,seq_len, num, ground_truth,test = False):
        load = dataset[dataset.columns[0]]
        feature_num = len(dataset.columns)
        load = load.tolist()
        ground_truth = ground_truth.values.tolist()
        dataset = dataset.values.tolist()

        seq = []
        for i in range(0, len(dataset) - seq_len - num + 1):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(1, feature_num):
                    #if c <= 2:
                    #    x.append(dataset[j][c])
                    #else:
                    x.append(dataset[j+1][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(ground_truth[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        seq = MyDataset(seq)
        if test is False:
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        else:
            seq = DataLoader(dataset=seq, shuffle=shuffle, num_workers=0, drop_last=True)
        return seq

    Dtr = []
    Val = []

    for i in range(len(train)):
        tr_seq = rolling_data(train[i], B, False,seq_len, num,tra_gt[i])
        Dtr.append(tr_seq)

    for j in range(len(val)):
        val_seq = rolling_data(val[j], B, False,seq_len, num,val_gt[j],test = True)
        Val.append(val_seq)

    Dte = rolling_data(test, B, False,seq_len, num,test_ground_truth,test = True)

    return Dtr, Val, Dte, m, n

def nn_seq_mo_s2s(seq_len, B, num,predict_index,removed_factors = None):
    print(predict_index)
    data = csv_reader("data/data_pp.csv",predict_index,removed_factors)

    min_date = data.index.min()
    split_date = min_date + pd.DateOffset(weeks=52 * 12)
    train_valid = data[data.index <= split_date]
    test = data[data.index > split_date]
    test_ground_truth = test.iloc[:,[0]]

    m, n = np.max(train_valid[train_valid.columns[0]]), np.min(train_valid[train_valid.columns[0]])


    # Min-Max Scaler
    columns = train_valid.columns

    scaler = MinMaxScaler()
    train_valid[columns] = scaler.fit_transform(train_valid[columns])
    test[columns] = scaler.transform(test[columns])

    train,val,tra_gt,val_gt = split_time_series(train_valid)

    def rolling_data(dataset, batch_size, shuffle,seq_len, num, ground_truth,test = False):
        load = dataset[dataset.columns[0]]
        feature_num = len(dataset.columns)
        load = load.tolist()
        ground_truth = ground_truth.values.tolist()
        dataset = dataset.values.tolist()

        seq = []
        for i in range(0, len(dataset) - seq_len - num + 1):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(1, feature_num):
                    #if c <= 2:
                    #    x.append(dataset[j][c])
                    #else:
                    x.append(dataset[j+1][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(ground_truth[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))
        seq = MyDataset(seq)
        if test is False:
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        else:
            seq = DataLoader(dataset=seq, shuffle=shuffle, num_workers=0, drop_last=True)
        return seq

    Dtr = []
    Val = []

    for i in range(len(train)):
        tr_seq = rolling_data(train[i], B, False,seq_len, num,tra_gt[i])
        Dtr.append(tr_seq)

    for j in range(len(val)):
        val_seq = rolling_data(val[j], B, False,seq_len, num,val_gt[j],test = True)
        Val.append(val_seq)

    Dte = rolling_data(test, B, False,seq_len, num,test_ground_truth,test = True)

    return Dtr, Val, Dte, m, n
