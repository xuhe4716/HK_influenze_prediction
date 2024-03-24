import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
def folder_creator(folder_path,file_name):
    full_path = os.path.join(folder_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return full_path

def set_lag(df,predict_index):
    virus_index = ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus']
    for virus in virus_index:
        if virus != predict_index:
            df[virus] = df[virus].shift(1)

    plt.figure(figsize=(10,4))
    plt.plot(df['date'],df[predict_index])
    plt.title(predict_index, fontsize=18)

    folder_path = 'result/{}'.format(predict_index)
    file_name = 'trend.png'
    full_path = folder_creator(folder_path,file_name)
    plt.savefig(full_path)

def acf(df,predict_index,lag = 52):
    plot_acf(df[predict_index], lags=lag)
    plt.title('Autocorrelation Function')

    folder_path = 'result/{}'.format(predict_index)
    file_name = 'acf_lag{}.png'.format(lag)
    full_path = folder_creator(folder_path,file_name)
    plt.savefig(full_path)

def pacf(df,predict_index,lag = 52):
    plot_pacf(df[predict_index], lags=lag)
    plt.title('Partial Autocorrelation Function')

    folder_path = 'result/{}'.format(predict_index)
    file_name = 'pacf_lag{}.png'.format(lag)
    full_path = folder_creator(folder_path,file_name)
    plt.savefig(full_path)

def correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Correlation Matrix')

    plt.savefig('result/correlation_matrix')

def main():
    for virus in ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus']:
        df = pd.read_csv('data_pp.csv')
        df['date']= pd.to_datetime(df['date'])
        df = df.set_index(df['date'])
        start = pd.to_datetime('2015-01-04')
        df = df[start:]

        predict_index = virus
        set_lag(df, predict_index)
        acf(df, predict_index)
        pacf(df, predict_index)
        correlation_matrix(df)


if __name__ == '__main__':
    main()