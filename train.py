# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
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

# parameters
curPath = os.path.abspath(os.path.dirname(__file__)) #获取当前文件路径
sys.path.append(curPath)
LSTM_PATH = curPath + '/mtl.pkl'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #选择驱动方式
gamma = 0.1
step_size = 12
batch_size = 36
lr = 0.006
num_layers = 1
hidden_size = 100
weight_decay = 1e-4
opti = 'adam'

# Multiple outputs LSTM


def main():
    #解析命令行参数和选项的标准模块
    parser = argparse.ArgumentParser() #创建一个解析对象

    parser.add_argument('--predict_index', type=str, default='pos_index', help='predict feature name')
    parser.add_argument('--epochs', type=int, default=30, help='epoch')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=6, help='seq len')
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    parser.add_argument('--model', type=str, default="Seq2Seq", help='LSTM direction')

    args = parser.parse_args()
    return args


def mo_val(model, Val):
    model.eval()
    loss_function = nn.L1Loss().to(device)
    model.eval()
    val_loss = []
    y = []
    pred = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
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
        if opti == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)
        elif opti == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=weight_decay)    #Optimizer
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  #Equal Interval Adjustment Learning Rate StepLR
    # training
    min_epochs = 10
    min_val_loss = 5

    epochs = args.epochs
    for epoch in tqdm(range(epochs)):
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
        val_loss,val_mape,val_r2 = mo_val(model, Val)
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

    #state = {'models': best_model.state_dict()}
    torch.save(best_model,'save.pt')  # save model
    if not isinstance(model, Baseline):
        plt.plot([i for i in range(0,epochs)],train_loss_big,label = 'Train')
        plt.plot([i for i in range(0,epochs)],val_loss_big, label = 'Val')
        plt.legend()
        plt.show()

        plt.plot([i for i in range(0,epochs)],val_r2_big,label = 'R2')
        plt.plot([i for i in range(0,epochs)],val_mape_big, label = 'MAPE')
        plt.legend()
        plt.show()

    return best_model




if __name__ == '__main__':
    args = main() # Parsing model parameters
    Dtr, Val, Dte, m, n =nn_seq_mo(seq_len=args.seq_len, B=batch_size, num=args.output_size,
                                   predict_index = args.predict_index,removed_factors = ['RSV','paraflu12', 'paraflu34'])

    if args.model == "BiLSTM":
        model = BiLSTM(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,device=device).to(device)
    elif args.model == 'LSTM':
        model = LSTM(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,device=device).to(device)
    elif args.model == 'Seq2Seq':
        model = Seq2Seq(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,device=device).to(device)
    elif args.model == "Baseline":
        model = Baseline(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size).to(device)
    #mo_train(args, model, Dtr[-1], Val[-1], LSTM_PATH)
    #mo_test(Dte, m, n)

    if args.model == "Baseline":
        mo_train(args, model, Dtr[-1], Val[-1], LSTM_PATH)
        #mo_test(Dte, m, n)
    
    else:
        for i in range(len(Dtr)):
            mo_train(args, model,Dtr[i], Val[i], LSTM_PATH)
            #mo_test(Dte, m, n)


