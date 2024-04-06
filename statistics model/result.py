from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

predict_index = "RSV"
model = "ARIMA"
dff = pd.read_csv(f"Result/ARIMA/{predict_index}.csv")

plt.figure(dpi=150,figsize=(7,4))
plt.plot(dff['pred'],label='Pred')
plt.plot(dff['actual'],label = 'True')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.AutoLocator())
# tick_spacing = 3
# plt.xticks(ticks=range(0, len(dff), tick_spacing), rotation=45)
plt.legend()
plt.savefig(f"Result/{model}/{predict_index}.png")

# metrics

def Evaluation_index(Y_test1,pre):
    from sklearn.metrics import r2_score, mean_squared_error,explained_variance_score,mean_absolute_error
    r2 = r2_score(Y_test1,pre)
    ev = explained_variance_score(Y_test1,pre)
    mse = mean_squared_error(Y_test1,pre)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test1,pre)

    pre = pre.reshape(-1)
    Y_test1 = Y_test1.reshape(-1)
    INDEX = []
    page = 0
    for i in Y_test1:
        if i ==0:
            INDEX.append(page)
        page +=1
    if INDEX !=[]:
        Y_test1 = np.delete(Y_test1,INDEX,0)
        pre     = np.delete(pre,INDEX,0)
    mape = (sum(abs((pre - Y_test1)/(Y_test1)))/len(Y_test1))
    print('r2:',r2)
    print('mse:',mse)
    print('rmse:',rmse)
    print('mae:',mae)
    print('mape:',mape)
# def z_score(data):
#     data = data.astype(float)
#     Mean = data.mean()
#     Var = ((data - Mean)**2).mean()
#     Std = pow(Var,0.5)
#     data = (data - Mean)/Std  # 标准化
#     return Mean,Std,data
# Mean,Std,_ = z_score(train)
# Evaluation_index((dff['真实值'].values - Mean)/Std,(predictions - Mean)/Std)
Evaluation_index(dff['actual'].values,np.array(dff['pred']))