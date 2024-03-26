import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import statsmodels.api as sm
import numpy as np

def folder_creator(folder_path,file_name):
    full_path = os.path.join(folder_path, file_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return full_path

def set_lag(df,predict_index):
    virus_index = ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus']
    #for virus in virus_index:
    #    if virus != predict_index:
    #        df[virus] = df[virus].shift(1)

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


def granger_causality(df, predict_index, factor_indices, max_lag=4):
    """
    Function to perform Granger causality test and plot the results for multiple predictors in a 2x2 layout.

    :param df: pandas DataFrame containing the time series data
    :param predict_index: index (column name) of the variable to be predicted (Y)
    :param factor_indices: list of indices (column names) of the predictor variables (Xs)
    :param max_lag: maximum lag to which the Granger causality test will be performed
    """
    # Calculate the number of rows and columns for the subplot
    num_factors = len(factor_indices)
    num_rows = 2
    num_cols = 2

    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8), constrained_layout=True)

    for idx, factor_index in enumerate(factor_indices):
        # Get the current axis for the subplot
        ax1 = axs[idx // num_cols, idx % num_cols]

        # Run the Granger causality test
        #test_result = sm.tsa.stattools.grangercausalitytests(np.column_stack((Y, X)), maxlag=max_lag, verbose=False)
        test_result = sm.tsa.stattools.grangercausalitytests(np.column_stack((df[predict_index], df[factor_index])), maxlag=max_lag, verbose=False)

        # Extract F-statistic and P-values for plotting
        f_stats = [test_result[i+1][0]['ssr_ftest'][0] for i in range(max_lag)]
        p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
        lags = range(1, max_lag + 1)


        # Plotting the results
        color = 'tab:red'
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('F-statistic', color=color)
        ax1.plot(lags, f_stats, color=color, marker='o', label='F-statistic')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f'{factor_index} causing {predict_index}')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('P-value', color=color)
        ax2.plot(lags, p_values, color=color, marker='x', linestyle='--', label='P-value')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=0.05, color='grey', linestyle='--')  # 显著性水平线





    # Hide any unused subplots
    for i in range(idx + 1, num_rows * num_cols):
        fig.delaxes(axs[i // num_cols, i % num_cols])
        #axs.legend(loc='upper right')
        #ax2.legend(loc='upper right')

    folder_path = 'result/{}'.format(predict_index)
    file_name = 'granger_causality_{}.png'.format(predict_index)
    full_path = folder_creator(folder_path,file_name)
    plt.savefig(full_path)



def main():
    granger_list = ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus']
    virus_ls = ['pos_index','RSV','paraflu12', 'paraflu34','Adenovirus']
    df = pd.read_csv('data_pp.csv')
    df['date']= pd.to_datetime(df['date'])
    df = df.set_index(df['date'])
    start = pd.to_datetime('2004-07-04')
    df = df[start:]
    for virus in virus_ls:
        predict_index = virus
        set_lag(df, predict_index)
        acf(df, predict_index)
        pacf(df, predict_index)
        granger_list.remove(virus)
        granger_causality(df, virus,granger_list)
        granger_list = virus_ls.copy()

    correlation_matrix(df)



if __name__ == '__main__':
    main()