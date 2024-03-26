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
from model.GRU import GRU
from model.Baseline import Baseline
from rolling import nn_seq_mo

def seed_everything(seed=999):
    import random
    random.seed(seed)
    import os
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(True)
        torch.mps.manual_seed(seed)


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
weight_decay = 0
opti = 'adam'

# Multiple outputs LSTM


def main():
    #解析命令行参数和选项的标准模块
    parser = argparse.ArgumentParser() #创建一个解析对象

    parser.add_argument('--predict_index', type=str, default='RSV', help='predict feature name') # 预测RSV时把这里的pos_index替换成RSV
    parser.add_argument('--epochs', type=int, default=40, help='epoch')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension') # 特征数量 11个特征加自己本身的上一个时刻的ground truth
    parser.add_argument('--seq_len', type=int, default=6, help='seq len') # 使用t - 5 - t时刻的数据
    parser.add_argument('--output_size', type=int, default=1, help='output dimension') # 默认输出t+1 - t+4的数据
    parser.add_argument('--predicted_size', type=int, default=1, help='predicted dimension') # only for seq2seq
    parser.add_argument('--model', type=str, default="GRU", help='model') # 换成Baseline可以看基模型

    args = parser.parse_args()
    #seed_everything()
    #args = main() # Parsing model parameters
    if args.model == "Seq2Seq":
        Dtr, Val, Dte, m, n =nn_seq_mo(seq_len=args.seq_len, B=batch_size, num=args.predicted_size,
                                       predict_index = args.predict_index,removed_factors = ["pos_index",'paraflu12','paraflu34']) #预测RSV时把这里的RSV 替换成pos_index
    else:
        Dtr, Val, Dte, m, n =nn_seq_mo(seq_len=args.seq_len, B=batch_size, num=args.output_size,
                                   predict_index = args.predict_index,removed_factors = ["pos_index",'paraflu12','paraflu34']) #预测RSV时把这里的RSV 替换成pos_index

    if args.model == "BiLSTM":
        model = BiLSTM(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,device=device).to(device)
    elif args.model == 'LSTM':
        model = LSTM(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,device=device).to(device)
    elif args.model == 'Seq2Seq':
        model = Seq2Seq(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,pred_size = args.predicted_size,device=device).to(device)
    elif args.model == 'GRU':
        model = GRU(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size,device=device).to(device)
    elif args.model == "Baseline":
        model = Baseline(args.input_size, hidden_size, num_layers, args.output_size, batch_size=batch_size).to(device)
    #mo_train(args, model, Dtr[-1], Val[-1], LSTM_PATH)
    #mo_test(Dte, m, n)
    val_mape = []
    val_r2 = []

    '''
    if args.model == "Baseline":
        mo_train(args, model, Dtr[-1], Val[-1], LSTM_PATH)
        #mo_test(Dte, m, n)

    else:
        for i in range(len(Dtr)):
            best_model,final_val_mape,final_val_r2 = mo_train(args, model,Dtr[i], Val[i], LSTM_PATH)
            torch.save(best_model,"save_model/{}_model.pt".format(i))
            val_mape.append(final_val_mape)
            val_r2.append(final_val_r2)

        plt.plot([i for i in range(len(Dtr))],val_mape,label = "mape")
        plt.plot([i for i in range(len(Dtr))],val_r2,label = 'r2')
        plt.xlabel(model)
        plt.legend()
        plt.show()
    '''
    best_model,final_val_mape,final_val_r2 = mo_train(args, model,Dtr[-1], Val[-1], LSTM_PATH)
    torch.save(best_model,"save_model/5_model.pt")


class LogCoshLoss(nn.Module):
    def __init__(self):
        """
        初始化Log-Cosh损失函数。
        """
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        计算Log-Cosh损失。
        :param y_pred: 预测值，形状为(batch_size, )。
        :param y_true: 真实值，形状为(batch_size, )。
        :return: Log-Cosh损失。
        """
        loss = torch.mean(torch.log(torch.cosh(y_pred - y_true)))
        return loss

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
            model.train()
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



    #state = {'models': best_model.state_dict()}
    #torch.save(best_model,'save.pt')  # save model
    if not isinstance(model, Baseline):
        plt.plot([i for i in range(0,epochs)],train_loss_big,label = 'Train')
        plt.plot([i for i in range(0,epochs)],val_loss_big, label = 'Val')
        plt.legend()
        plt.show()

        plt.plot([i for i in range(0,epochs)],val_r2_big,label = 'R2')
        plt.plot([i for i in range(0,epochs)],val_mape_big, label = 'MAPE')
        plt.legend()
        plt.show()

    print("Predcting....")
    _,final_val_mape,final_val_r2 = mo_val(best_model, Val)
    print("Validation set result---")
    print("val_mape {:.8f} val_r2 {:.8f}".format(final_val_mape,final_val_r2))

    return best_model,final_val_mape,final_val_r2




if __name__ == '__main__':
    main()

