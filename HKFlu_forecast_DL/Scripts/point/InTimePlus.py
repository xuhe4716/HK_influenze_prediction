"""
InTimePlus model train and predict, based on tsai package
Yearly rolling 
Author: Du, He
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import random
import torch
import copy
import os
import sys
sys.path.append(".")
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from Model.LstmModel import LstmDataset
from data import DataTool
import joblib
import optuna
from tsai.all import TimeSplitter, TSForecasting, TSStandardize, TSForecaster, mse, ShowGraph
from fastai.callback.tracker import SaveModelCallback
import argparse
############################################### data preparation ##########################################
mode = 'test8'
test_start_date = '2007-11-01'
test_end_date = None#'2016-06-30'
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='weather_data', help='data folder')
parser.add_argument('--predict_index', type=str, default='ILI', help='predict feature name')
parser.add_argument('--columns', nargs='+', help='predict columns')

args = parser.parse_args()
data_folder = args.data
predict_index = args.predict_index
col_list = args.columns
start_time = datetime.now()
pwd=os.path.abspath(os.path.dirname(os.getcwd()))
origin_path = pwd#os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
print("pwd = ", pwd, ", origin_path = ", origin_path)
path = f'{data_folder}/{predict_index}.csv'
#col_list = ['temp.max','temp.min', 'relative.humidity',
#            'total.rainfall', 'solar.radiation',
#            'monthid', 'weekid', 'rate']
dr = DataTool()
df_o = dr.data_output(path, col_list, mode = 'log')
df = copy.deepcopy(df_o)
model_name = f'InTimePlus_v3_nontuning_rolling_v2_{predict_index}_{data_folder.split("/")[1]}'

# df = df.loc[df.index>pd.to_datetime('2009-12-31'),:]
# print("finally, start date = ", df.index.min())

############################################### modeling ##########################################
study = joblib.load("./model_hyperparam/Separate_InTimePlus.pkl") #load
trial = study.best_trial
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

batch_size1=trial.params['batch_size1']
num_filters1=trial.params['num_filters1']
depth1 = trial.params['depth1']
dropout_rate1= trial.params['dropout_rate1']
learning_rate1 = trial.params['learning_rate1']

batch_size2=trial.params['batch_size2']
num_filters2=trial.params['num_filters2']
depth2 = trial.params['depth2']
dropout_rate2= trial.params['dropout_rate2']
learning_rate2 = trial.params['learning_rate2']

seq_length = 14
pred_stamp = 9

def one_bootstrap(i, df, test_size, total_pred_horizon):
    torch.manual_seed(i)
    random.seed(i)
    np.random.seed(i)
    print("---- test size = ", test_size, ", total prediction horizon = ", total_pred_horizon)
    # week 1
    pred_stamp = 1
    data_deal1 = LstmDataset(sequence_length=seq_length, 
                        batch_size=0, 
                        pred_stamp=pred_stamp)
    train_datadict1, _, test_datadict1 = data_deal1.get_train_val_test_dataset(df, test_size = test_size, sample_rate = None, validation=False)
    rate_max1, rate_min1 = data_deal1.get_rate_scaler()

    x_train1, y_train1= train_datadict1['x_data'].permute(0,2,1), train_datadict1['y_data']
    x_test1, y_test1= test_datadict1['x_data'].permute(0,2,1), test_datadict1['y_data']
    print("11111 x shape=", x_train1.shape,"y shape", y_train1.shape)
    print("x shape=", x_train1.shape,"y shape", y_train1.shape)
    val_size = np.random.choice(np.array(range(78,104)),size = 1)[0]
    splits1 = TimeSplitter(valid_size=val_size, fcst_horizon=pred_stamp, show_plot=False)(y_train1) 
    print("--- week 1 split ----")
    print(x_train1[splits1[0]].shape, y_train1[splits1[0]].shape)
    print(x_train1[splits1[1]].shape, y_train1[splits1[1]].shape)
    print(x_test1.shape, y_test1.shape)
    fcst1 = TSForecaster(x_train1, y_train1, splits=splits1, bs=batch_size1, arch='InceptionTimePlus', 
                         arch_config={'nf':num_filters1, 'fc_dropout':dropout_rate1, 'depth':depth1},
                        metrics=mse, train_metrics=True, seed=i, verbose=False)
    fcst1.fit_one_cycle(50,lr_max=learning_rate1,cbs=[SaveModelCallback(monitor='valid_loss')])
    raw_preds, target, preds1 = fcst1.get_X_preds(x_test1)

    # week after 1
    pred_stamp = total_pred_horizon - 1
    data_deal2 = LstmDataset(sequence_length=seq_length, 
                        batch_size=0, 
                        pred_stamp=pred_stamp)
    train_datadict2, _, test_datadict2 = data_deal2.get_train_val_test_dataset(df, test_size = test_size, sample_rate = None, validation=False)
    rate_max2, rate_min2 = data_deal2.get_rate_scaler()

    x_train2, y_train2= train_datadict2['x_data'].permute(0,2,1), train_datadict2['y_data']
    x_test2, y_test2= test_datadict2['x_data'].permute(0,2,1), test_datadict2['y_data']
    x_test2, y_test2 = x_test2[1:,:,:], y_test2[1:,:]
    print("11111 x shape=", x_train2.shape,"y shape", y_train2.shape)
    print("x shape=", x_train2.shape,"y shape", y_train2.shape)
    splits2 = TimeSplitter(valid_size=val_size, fcst_horizon=pred_stamp, show_plot=False)(y_train2) 
    print("--- week after 1, split ----")
    print(x_train2[splits2[0]].shape, y_train2[splits2[0]].shape)
    print(x_train2[splits2[1]].shape, y_train2[splits2[1]].shape)
    print(x_test2.shape, y_test2.shape)
    # 替换rate为预估值
    for i in range(x_test2.shape[0]):
        x_test2[i,-1,-1] = preds1[i]
    print(x_test2.shape, y_test2.shape)
    fcst2 = TSForecaster(x_train2, y_train2, splits=splits2, bs=batch_size2, arch='InceptionTimePlus', 
                         arch_config={'nf':num_filters2, 'fc_dropout':dropout_rate2, 'depth':depth2},
                        metrics=mse, train_metrics=True, seed=i, verbose=False)
    fcst2.fit_one_cycle(50,lr_max=learning_rate2,cbs=[SaveModelCallback(monitor='valid_loss')])
    raw_preds, target, preds2 = fcst2.get_X_preds(x_test2)

    preds1 = preds1*(rate_max1 -  rate_min1)+rate_min1
    preds1 = np.exp(preds1)
    preds2 = preds2*(rate_max2 - rate_min2)+rate_min2
    preds2 = np.exp(preds2)
    print("pred shape = ", preds1.shape, preds2.shape)
    return preds1, preds2

def one_rolling(df, test_start, test_end, pred_horizon, exp_mode = True, bootstrap_times = 100, random_state = None):
    df_t = copy.deepcopy(df.loc[df.index<=pd.to_datetime(test_end),:])
    test_size = df_t.loc[df_t.index > test_start,:].shape[0]
    # print("---------------------------- test_start = ", test_start, ', test_end = ',test_end,', df_t shape = ',df_t.shape,', test_size = ', test_size)
    df_test = copy.deepcopy(df_t.loc[df_t.index >= test_start,:])
    re_test = dr.origin_re_output(df_test, left_len=0, pred_len = pred_horizon, exp_mode=exp_mode)

    for bst in range(bootstrap_times):
        print("-------------------------- bootstrap = ", bst, " ----------------------------------")
        seed_ = random_state if random_state is not None else bst
        y_pred_i1,y_pred_i4 = one_bootstrap(seed_, df_t, test_size, total_pred_horizon=pred_stamp)
        # print("pred_shape = ",y_pred_i.shape)
        re_test_pred = pd.DataFrame(y_pred_i1[0:y_pred_i4.shape[0]], columns = [f'boot_{bst}'])
        re_test_pred['week_ahead'] = 0
        for i in range(pred_horizon-1):
            re_t = pd.DataFrame(y_pred_i4[:,i], columns = [f'boot_{bst}'])
            re_t['week_ahead'] = i+1
            re_test_pred = pd.concat([re_test_pred, re_t], ignore_index=True)
        re_test = pd.concat([re_test, re_test_pred[[f'boot_{bst}']]], axis=1)
    
    return re_test

test_start_date = pd.to_datetime('2009-11-01')
max_year_range = 20
year_step = 1
rolling_dates = [test_start_date + timedelta(days = 52 * 7 * i) for i in range(0,max_year_range,year_step) if test_start_date + timedelta(days = 52 * 7 * i) < (pd.to_datetime('2019-10-30')-timedelta(days = (pred_stamp-1) * 7))]
rolling_dates.append(pd.to_datetime('2019-10-30')-timedelta(days = (pred_stamp-1) * 7))

df_test_total = copy.deepcopy(df.loc[df.index >= test_start_date,:])
re_test_total = pd.DataFrame()

for i_date in range(len(rolling_dates)-1):
    test_start, test_end = rolling_dates[i_date], rolling_dates[i_date+1]+timedelta(days = (pred_stamp-1) * 7)
    print("----------------------------- i_date = ", i_date,", test_start = ", test_start, ', test_end = ',test_end)
    re_test = one_rolling(df, test_start, test_end, pred_stamp, bootstrap_times = 1, random_state=i_date)
    re_test_total = pd.concat([re_test_total, re_test], axis=0)

############################################### save ##########################################
re_test_total.rename(columns={'boot_0':'point'}, inplace=True)
re_test_total['point_avg'] = re_test_total['point']
dr.point_write(re = re_test_total, origin_path=origin_path, mode = mode, model_name=model_name)

end_time = datetime.now()
print("at the time<",start_time.strftime('%Y-%m-%d %H:%M:%S'),">, ",model_name," begin,"," at the time<",end_time.strftime('%Y-%m-%d %H:%M:%S'),"> finished.") 
print("The running time totally =", (end_time-start_time).seconds," seconds.")      

