import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 加载数据
file_path = '天气720.csv'
weather_data = pd.read_csv(file_path)

# 前几行
weather_data.head()

# 将“ds”转换为日期时间并将其设置为索引
weather_data['ds'] = pd.to_datetime(weather_data['ds'])
weather_data.set_index('ds', inplace=True)

# 检查是否有缺失值
if weather_data.isnull().values.any():
    print("存在则均值填充")
    weather_data.fillna(weather_data.mean(), inplace=True)

# 划分训练测试集
# train_size = int(len(weather_data) * 0.8)
# train, test = weather_data.iloc[:train_size], weather_data.iloc[train_size:]

train, test = weather_data.iloc[:710], weather_data.iloc[710:]
print(train)

# DW检验
dw_statistic = durbin_watson(train)
print("DW Statistic:", dw_statistic)

# 检查时序数据的平稳性
adf_test = adfuller(train)
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])
print("Critical Values:", adf_test[4])

# 自相关和偏自相关图
plot_acf(train, lags=40)
plt.title("ACF")
plt.show()

plot_pacf(train, lags=40)
plt.title("PACF")
plt.show()

# 对数据进行一次差分
data_diff = train.diff().dropna()
plt.plot(data_diff)
plt.show()

# 一次差分后的平稳性
adf_test_diff = adfuller(data_diff)
print("ADF Statistic (diff):", adf_test_diff[0])
print("p-value (diff):", adf_test_diff[1])
print("Critical Values (diff):", adf_test_diff[4])

# 自相关和偏自相关图
plot_acf(data_diff, lags=40)
plt.title("ACF")
plt.show()

plot_pacf(data_diff, lags=40)
plt.title("PACF")
plt.show()

# 定义ARIMA模型的参数范围
p_range = range(0, 3)
d_range = range(0, 1)
q_range = range(0, 3)

# 寻找最佳ARIMA模型
best_aic = float("inf")
best_model = None
for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                print(f"ARIMA({p}, {d}, {q}) - AIC: {aic}")
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_model = model_fit
            except Exception as e:
                print(f"ARIMA({p}, {d}, {q}) - Error: {e}")

# 输出最佳模型的信息
print(f"Best ARIMA Model: ARIMA{best_order} - AIC: {best_aic}")
print(best_model.summary())

# 残差分析
residuals = best_model.resid

# 残差图
plt.figure(figsize=(6, 4))
plt.plot(residuals)
plt.title('Residuals of Best ARIMA Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

# 残差的ACF和PACF图
plot_acf(residuals, lags=20)
plt.title('ACF of Residuals')
plt.show()

plot_pacf(residuals, lags=20)
plt.title('PACF of Residuals')
plt.show()

# 进行LB检验
lb_test = acorr_ljungbox(residuals, lags=40, return_df=True)

# 绘制LB统计量的p值变化图
plt.figure(figsize=(6, 4))
plt.scatter(range(len(lb_test['lb_pvalue'])), lb_test['lb_pvalue'])
plt.title('P-values for Ljung-Box Statistic')
plt.xlabel('Lags')
plt.ylabel('p-value')
plt.axhline(y=0.05, color='r', linestyle='--')  # 显著性水平线
plt.show()

# 模型拟合与预测
pred = best_model.forecast(steps=len(test))
print(pred)

# 评估
mse = mean_squared_error(test, pred)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
print('MSE:', mse)
print('mae:', mean_absolute_error(test, pred))
print('r2:', r2_score(test, pred))

# 可视化
plt.figure(figsize=(8, 5))
plt.plot(train.index, train, label='Actual_train', color='y')
plt.plot(test.index, test, label='Actual', color='b')
plt.plot(test.index, pred, label='Forecast', color='red', linestyle='--')
plt.title('Weather Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 拟合最佳ARIMA模型到整个数据集
best_model_full = ARIMA(weather_data, order=best_order).fit()

# 预测步数
steps = 10
# 进行预测
predictions = best_model_full.forecast(steps=steps)
print(predictions)

# 绘制历史数据
plt.figure(figsize=(6, 4))
# 绘制预测数据
plt.plot(range(len(weather_data)), weather_data)
plt.plot(range(len(weather_data), len(weather_data)+steps), predictions, label='Forecast')
# 添加标题和标签
plt.title('Historical Data and Forecast')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
# 显示图表
plt.show()

