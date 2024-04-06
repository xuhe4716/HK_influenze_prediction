from rolling_stat import arima_seq_mo
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
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

name = "RSV"
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
dff.to_csv(f'Result/ARIMA/{name}.csv',index = False)
print(dff)


