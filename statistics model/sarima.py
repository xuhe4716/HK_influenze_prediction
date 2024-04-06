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
Dtv = Dtv.diff(52).dropna()


# set p,d,q range
p_range = q_range  = P_range  = Q_range  = range(0, 3)  # AR, I, MA参数从0到2
d_range  = [0]
D_range  = [1]
s = 52
#parameters = list(product(p, d, q))
#parameters = [(p,d,q) for (p,d,q) in parameters]
#print(parameters)

# best p,d,q search
best_aic = float("inf")
best_model = None
best_order = None
best_seasonal_order = None
results = []

for p in p_range:
    for d in d_range:
        for q in q_range:
            for P in P_range:
                for D in D_range:
                    for Q in Q_range:
                        try:
                            # 注意：这里是SARIMAX，而不是ARIMA
                            model = SARIMAX(Dtv, order=(p, d, q), seasonal_order=(P, D, Q, s))
                            model_fit = model.fit(disp=0)
                            aic = model_fit.aic
                            print(f"SARIMA({p}, {d}, {q})x({P},{D},{Q},{s}) - AIC: {aic}")
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                best_seasonal_order = (P, D, Q, s)
                                best_model = model_fit
                        except Exception as e:
                            continue

print(f"Best parameter is {best_order} {best_seasonal_order}. \n Best aic is {best_aic}.")
model_fit = best_model
print(f"Best model: {model_fit.summary()}")

Dtv = list(Dtv.values)
Dte = list(Dte.values)
predictions = []
# Predict
for t in range(len(Dte)):
    model = SARIMAX(Dtv, order=best_order, seasonal_order=best_seasonal_order)
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
dff.to_csv(f'Result/SARIMA/{name}.csv',index = False)
print(dff)


