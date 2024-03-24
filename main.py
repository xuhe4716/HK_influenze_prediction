# -*- coding:utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import copy
from itertools import chain
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from model.LSTM import LSTM, BiLSTM
from model.Seq2Seq import Seq2Seq
from model.Baseline import Baseline
from rolling import nn_seq_mo

curPath = os.path.abspath(os.path.dirname(__file__)) #获取当前文件路径
sys.path.append(curPath)
LSTM_PATH = curPath + '/mtl.pkl'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #选择驱动方式

# Multiple outputs LSTM
def mo_args_parser():  #解析命令行参数和选项的标准模块
    parser = argparse.ArgumentParser() #创建一个解析对象

    parser.add_argument('--predict_index', type=str, default='pos_index', help='predict feature name')
    parser.add_argument('--epochs', type=int, default=30, help='epoch')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=6, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='num layers')
    parser.add_argument('--lr', type=float, default=0.006, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=36, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--model', type=str, default="Seq2Seq", help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=12, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')

    args = parser.parse_args()
    return args


def mo_val(args, model, Val):
    model.eval()
    loss_function = nn.L1Loss().to(args.device)
    model.eval()
    val_loss = []
    y = []
    pred = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            y.extend(label.to(torch.device('cpu')))
            pred.extend(y_pred.to(torch.device('cpu')))
            val_loss.append(loss.item())
    val_mape = mean_absolute_percentage_error(y, pred)
    R2 = r2_score(y, pred)
    return np.mean(val_loss), val_mape,R2


def mo_train(args, model, Dtr, Val, path):
    best_model = None

    train_loss_big = []
    val_loss_big = []
    val_r2_big = []
    val_mape_big = []



    loss_function = nn.L1Loss().to(device)   #损失函数方式
    if not isinstance(model, Baseline):
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                        momentum=0.9, weight_decay=args.weight_decay)    #Optimizer
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  #Equal Interval Adjustment Learning Rate StepLR
    # training
    min_epochs = 10
    min_val_loss = 5


    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            if not isinstance(model, Baseline):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if not isinstance(model, Baseline):
            scheduler.step()

            # validation
        val_loss,val_mape,val_r2 = mo_val(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        if not isinstance(model, Baseline):
            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}\n val_mape {:.8f} val_r2 {:.8f}'.format(epoch, np.mean(train_loss), val_loss, val_mape,val_r2))
            train_loss_big.append(np.mean(train_loss))
            val_loss_big.append(val_loss)
            val_mape_big.append(val_mape)
            val_r2_big.append(val_r2)

        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)  # save model
    if not isinstance(model, Baseline):
        plt.plot([i for i in range(0,epoch+1)],train_loss_big,label = 'Train')
        plt.plot([i for i in range(0,epoch+1)],val_loss_big, label = 'Val')
        plt.legend()
        plt.show()

        plt.plot([i for i in range(0,epoch+1)],val_r2_big,label = 'R2')
        plt.plot([i for i in range(0,epoch+1)],val_mape_big, label = 'MAPE')
        plt.legend()
        plt.show()

    return best_model

def plot(y, pred,num_row):
    # plot
    x = [i for i in range(1, num_row + 1)]
    # print(len(y))
    x_smooth = np.linspace(np.min(x), np.max(x), 500)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    y_smooth = make_interp_spline(x, pred)(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.title('multiple_outputs')
    plt.show()

def mo_test(args, Dte, path, m, n):
    model.load_state_dict(torch.load(path)['models'])
    print('predicting...')
    model.eval()
    y = []
    pred = []
    for (seq, label) in Dte:
        with torch.no_grad():
            seq = seq.to(args.device)
            y.extend(label)
            y_pred = model(seq).to(torch.device('cpu'))
            pred.extend(y_pred)

    for tensor in pred:
        tensor[:] = (m - n) * tensor + n

    num_row_y = len(y)

    MSE_l = mean_squared_error(y, pred, multioutput='uniform_average')
    MAE_l = mean_absolute_error(y, pred)
    MAPE_l = mean_absolute_percentage_error(y, pred)
    R2 = r2_score(y, pred)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)

    plot(y, pred,num_row_y)


if __name__ == '__main__':
    args = mo_args_parser() # Parsing model parameters
    Dtr, Val, Dte, m, n =nn_seq_mo(seq_len=args.seq_len, B=args.batch_size, num=args.output_size,predict_index = args.predict_index,removed_factors = ['RSV','paraflu12', 'paraflu34'])

    if args.model == "BiLSTM":
        model = BiLSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, batch_size=args.batch_size,device=device).to(device)
    elif args.model == 'LSTM':
        model = LSTM(args.input_size, args.hidden_size, args.num_layers, args.output_size, batch_size=args.batch_size,device=device).to(device)
    elif args.model == 'Seq2Seq':
        model = Seq2Seq(args.input_size, args.hidden_size, args.num_layers, args.output_size, batch_size=args.batch_size,device=device).to(device)
    elif args.model == "Baseline":
        model = Baseline(args.input_size, args.hidden_size, args.num_layers, args.output_size, batch_size=args.batch_size).to(device)

    mo_train(args, model, Dtr[-1], Val[-1], LSTM_PATH)
    mo_test(args, Dte, LSTM_PATH, m, n)
    '''
    if args.model == "Baseline":
        mo_train(args, Dtr[-1], Val[-1], LSTM_PATH, args.model)
        mo_test(args, Dte, LSTM_PATH, m, n, args.model)
    else:
        for i in range(len(Dtr)):
            mo_train(args, Dtr[i], Val[i], LSTM_PATH, args.model)
            mo_test(args, Dte, LSTM_PATH, m, n, args.model)
    '''
