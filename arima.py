from rolling import arima_seq_mo
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
from itertools import product
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息，用于清理输出

import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as stats

name = "pos_index"
# Load
_, _, Dte,Dtv =arima_seq_mo(predict_index = name,removed_factors = ['weekid','yearid','monthid'])

# acf
plot_acf(Dtv)
plt.show()

# pacf
plot_pacf(Dtv)
plt.show()

# adf_test
def adf_test(series, signif=0.05):
    result = adfuller(series, autolag='AIC')
    pvalue = result[1]
    return pvalue

pvalue1 = adf_test(Dtv)
print('The data is stationary: ',pvalue1< 0.05)

# set p,d,q range
p = q = range(0, 3)  # AR, I, MA参数从0到2
d = [0]
parameters = list(product(p, d, q))
parameters = [(p,d,q) for (p,d,q) in parameters]
print(parameters)

# best p,d,q search
best_aic_bic = float("inf")
best_params = None
results = []

for param in parameters:
    try:
        model = ARIMA(Dtv, order=param)
        model_fit = model.fit()
        aic = model_fit.aic
        bic = model_fit.bic
        results.append((param, aic))
        if np.mean([aic,bic]) < best_aic_bic:
            best_aic_bic = aic
            best_params = param
    except:
        continue

print(f"Best parameter is {best_params}. \n Best aic bic is {best_aic_bic}.")
model = ARIMA(Dtv, order=best_params)
model_fit = model.fit()
print(f"Best model: {model_fit.summary()}")

Dtv = list(Dtv.values)
Dte = list(Dte.values)
predictions = []
# Predict
for t in range(len(Dte)):
    model = ARIMA(Dtv, order=best_params)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = Dte[t]
    Dtv.append(obs)

dff = pd.DataFrame()
dff['pred'] = predictions
dff['actual'] = Dte
#dff.index = df.iloc[-len(predictions):,0]
print(dff)

plt.figure(dpi=150,figsize=(7,4))
plt.plot(dff['pred'],label='Pred')
plt.plot(dff['actual'],label = 'True')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(plt.AutoLocator())
# tick_spacing = 3
# plt.xticks(ticks=range(0, len(dff), tick_spacing), rotation=45)
plt.legend()
plt.show()

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
Evaluation_index(dff['actual'].values,np.array(predictions))
