import warnings
warnings.filterwarnings('ignore')
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
import math
import numpy as np
from scipy.interpolate import make_interp_spline
import torch
from rolling import nn_seq_mo


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #选择驱动方式
batch_size = 36

def main():
    #解析命令行参数和选项的标准模块
    parser = argparse.ArgumentParser() #创建一个解析对象

    parser.add_argument('--model_path', type=str, default='save_model/5_model.pt', help='predict feature name')
    parser.add_argument('--predict_index', type=str, default='RSV', help='predict feature name') # 预测RSV时把这里的pos_index替换成RSV
    #parser.add_argument('--epochs', type=int, default=35, help='epoch')
    parser.add_argument('--input_size', type=int, default=12, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=6, help='seq len')
    parser.add_argument('--output_size', type=int, default=4, help='output dimension')
    args = parser.parse_args() # Parsing model parameters

    _, _, Dte, m, n =nn_seq_mo(seq_len=args.seq_len, B=batch_size, num=args.output_size,
                               predict_index = args.predict_index,removed_factors = ["pos_index",'paraflu12','paraflu34']) #预测RSV时把这里的RSV 替换成pos_index

    mo_test(args.model_path,Dte, m, n)

    return args


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

def mo_test(model_path, Dte, m, n):
    model = torch.load(model_path)
    #print(model)
    #model.load_state_dict(torch.load(path)['models'])

    print('predicting...')
    model.eval()
    y = []
    pred = []
    for (seq, label) in Dte:
        with torch.no_grad():
            seq = seq.to(device)
            y.extend(label)
            y_pred = model(seq).to(torch.device('cpu'))
            pred.extend(y_pred)

    for tensor in pred:
        tensor[:] = (m - n) * tensor + n

    #y = y[:-1]
    #pred = pred[1:]
    num_row_y = len(y)

    MSE_l = mean_squared_error(y, pred, multioutput='uniform_average')
    MAE_l = mean_absolute_error(y, pred)
    MAPE_l = mean_absolute_percentage_error(y, pred)
    R2 = r2_score(y, pred)

    RMSE  = math.sqrt(MSE_l)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)
    print('RMSE=%s'%RMSE)

    plot(y, pred,num_row_y)


if __name__ == '__main__':
    main()