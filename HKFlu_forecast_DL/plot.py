"""
    plot figure
    Richael-2023/6/16
    --------------------
    plot different figure
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import copy
import argparse
import csv
import os


class Plot_():
    def __init__(self):
        pass 

    def get_metric(self, df, pred_stamp = 4, log = False):
        df_plot1 = df.set_index('date', drop = True, inplace=False)
        smape_t, mape_t, rmse_t, mae_t = [], [], [], []
        for i in range(pred_stamp):
            df_t = df_plot1.loc[df_plot1['week_ahead'] == i,:][['true','pred']]
            if log == True:
                df_t = np.exp(df_t['pred']).applymap(np.exp)
            df_t = df_t.dropna()
            smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
            rmse_ = round(np.sqrt(np.mean((df_t['true'].values - df_t['pred'].values)**2)), 2)
            mape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values) / df_t['true'].values)), 2)
            mae_ = round(np.mean(np.abs(df_t['true'].values - df_t['pred'].values)), 2)
            print(f"week{i} predict SMAPE = ", smape_, "RMSE = ", rmse_, "MAPE = ", mape_, "MAE = ", mae_)
            smape_t.append(smape_)
            rmse_t.append(rmse_)
            mape_t.append(mape_)
            mae_t.append(mae_)
        return smape_t, rmse_t,mape_t,mae_t

    def get_plot(self, df, pred_stamp = 4, log = True, figsize = None):
        # df_plot1 = df.set_index('date', drop = True, inplace=False)
        if figsize is None:
            plt.figure(figsize=(14,23))
        else:
            plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace =0, hspace = 0.3)#调整子图间距
        for i in range(pred_stamp):
            df_plot1 = df.loc[df['week_ahead'] == i,['true','pred','date']]
            df_t = df_plot1.set_index('date', drop = True, inplace=False)
            if log == True:
                df_t = df_t.applymap(np.exp)
            smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
            rmse_ = round(np.sqrt(np.mean((df_t['true'].values - df_t['pred'].values)**2)), 2)
            print(f"week{i} predict SMAPE = ", smape_, "RMSE = ", rmse_)
            plt.subplot(pred_stamp,1,(i+1))
            plt.plot(df_t.index, df_t.true, color='blue')
            plt.plot(df_t.index, df_t.pred, color = 'orange')
            plt.legend(labels = ['true','pred'])
            plt.title(f'{i} Week Ahead: Prediction of Rate. SMAPE = {smape_}, RMSE = {rmse_}')

        plt.show()

    # def get_saved_plot(self, df, pred_stamp = 4, name = 'temp'):
    #     df_plot1 = df.set_index('date', drop = True, inplace=False)
    #     plt.figure(figsize=(14,23))
    #     plt.subplots_adjust(wspace =0, hspace = 0.3)#调整子图间距
    #     for i in range(pred_stamp):
    #         df_t = df_plot1.loc[df_plot1['week_ahead'] == i,:][['true','pred']]
    #         df_t = df_t.applymap(np.exp)
    #         smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
    #         print(f"week{i} predict MAPE = ", smape_)
    #         plt.subplot(pred_stamp,1,(i+1))
    #         plt.plot(df_t.index, df_t.true, color='blue')
    #         plt.plot(df_t.index, df_t.pred, color = 'orange')
    #         plt.legend(labels = ['true','pred'])
    #         plt.title(f'{i} Week Ahead: Prediction of Rate. SMAPE = {smape_}')

    #     fig_path = f'/Users/hkuph/richael/flu/FluForecasting/figure/{name}.png'
    #     plt.savefig(fig_path, dpi=450, bbox_inches='tight', facecolor='w')
    #     print(f"the figure has been saved as {name}")
    #     plt.show()
    def get_saved_plot(self, df, pred_stamp = 4, log = False, figsize = None, path = None, show = False):
        # df_plot1 = df.set_index('date', drop = True, inplace=False)
        if figsize is None:
            plt.figure(figsize=(14,23))
        else:
            plt.figure(figsize=figsize)
        plt.subplots_adjust(wspace =0, hspace = 0.3)#调整子图间距
        for i in range(pred_stamp):
            df_plot1 = df.loc[df['week_ahead'] == i,['true','pred','date']]
            df_t = df_plot1.set_index('date', drop = True, inplace=False)
            if log == True:
                df_t = df_t.applymap(np.exp)
            smape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values)/(np.abs(df_t['true'].values)+np.abs(df_t['pred'].values)))), 2)
            rmse_ = round(np.sqrt(np.mean((df_t['true'].values - df_t['pred'].values)**2)), 2)
            mape_ = round(np.mean(np.abs((df_t['true'].values - df_t['pred'].values) / df_t['true'].values)), 2)
            mae_ = round(np.mean(np.abs(df_t['true'].values - df_t['pred'].values)), 2)

            print(f"week{i} predict SMAPE = ", smape_, "RMSE = ", rmse_)
            plt.subplot(pred_stamp,1,(i+1))
            plt.plot(df_t.index, df_t.true, color='blue')
            plt.plot(df_t.index, df_t.pred, color = 'orange')
            plt.legend(labels = ['true','pred'])
            plt.title(f'{i} Week Ahead: Prediction of Rate. SMAPE = {smape_}, RMSE = {rmse_}')

        if path is None:
            fig_path = f'/Users/hkuph/richael/flu/FluForecasting/figure/fig_tmp.png'
        else:
            fig_path = path
        plt.savefig(fig_path, dpi=450, bbox_inches='tight', facecolor='w')
        if show == True:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='weather_data', help='data folder')
    parser.add_argument('--predict_index', type=str, default='ILI', help='predict feature name')
    parser.add_argument('--model', type=str, default='Baseline', help='predict feature name')
    args = parser.parse_args()

    data_folder = args.data.split("/")[1]
    print("====================================================================")
    print("data_folder",data_folder)
    predict_index = args.predict_index
    print("predict_index",predict_index)
    model = args.model
    print("model",model)

    df = pd.read_csv(f"Results/Point/forecast_{model}_v3_nontuning_rolling_v2_{predict_index}_{data_folder}_test8.csv")
    df['pred'] = df['point']
    p = Plot_()
    pred_stamp = 9
    smape_t, rmse_t,mape_t,mae_t = p.get_metric(df, pred_stamp = pred_stamp)

    folder_path_figure = f'Results/Result_figures/{data_folder}'
    if not os.path.exists(folder_path_figure):
        os.makedirs(folder_path_figure)
    p.get_saved_plot(df, pred_stamp = pred_stamp,path = f"{folder_path_figure}/{predict_index}_{model}.png")

    folder_path_metrics = f"Results/Result_metrics/{data_folder}"
    if not os.path.exists(folder_path_metrics):
        os.makedirs(folder_path_metrics)
    for i in range(pred_stamp):
        info = [predict_index,model,i, rmse_t[i],mape_t[i], smape_t[i], mae_t[i]]
        with open(f'{folder_path_metrics}/rolling_result_{predict_index}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(info)
    

