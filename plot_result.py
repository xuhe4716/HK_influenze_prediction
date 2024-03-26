import pandas as pd
import matplotlib.pyplot as plt
pos_index_data = {
    'model': ['Baseline', 'Seq2Seq', 'GRU', 'LSTM', 'Baseline', 'SeqtoSeq', 'GRU', 'LSTM'],
    'input_size': [6, 6, 6, 6, 6, 6, 6, 6],
    'output_size': [1, 1, 1, 1, 4, 4, 4, 4],
    'Mape': [0.292, 0.294, 0.291, 0.312, 0.570, 0.542, 0.530, 0.535],
    'R2': [0.856, 0.886, 0.891, 0.887, 0.430, 0.638, 0.621, 0.612],
    'MSE': [3.884, 3.091, 2.929, 3.05, 15.622921, 9.92, 10.39, 10.627],
    'MAE': [1.07, 0.96, 0.944, 0.959, 2.0734224, 1.663, 1.69, 1.69],
    'RMSE': [1.970, 1.758, 1.711, 1.746, 3.9525, 3.15, 3.22, 3.25]
}

pos_index_df = pd.DataFrame(pos_index_data)

# 绘制pos_index数据的图表
fig, axes = plt.subplots(5, 1, figsize=(10, 20))
metrics = ['Mape', 'R2', 'MSE', 'MAE', 'RMSE']

for ax, metric in zip(axes, metrics):
    for output_size in pos_index_df['output_size'].unique():
        subset = pos_index_df[pos_index_df['output_size'] == output_size]
        ax.plot(subset['model'], subset[metric], marker='o', label=f'Output Size={output_size}')
    ax.set_ylabel(metric)
    ax.legend()
    ax.set_title(f'pos_index Performance: {metric}')

plt.tight_layout()
plt.savefig("pos_index.png")

import matplotlib.pyplot as plt
import pandas as pd

# 创建RSV数据的DataFrame
rsv_data = {
    'model': ['Baseline', 'Seq2Seq', 'GRU', 'LSTM', 'Baseline', 'SeqtoSeq', 'GRU', 'LSTM'],
    'input_size': [6, 6, 6, 6, 6, 6, 6, 6],
    'output_size': [1, 1, 1, 1, 4, 4, 4, 4],
    'Mape': [0.273, 0.272, 0.279, 0.276, 0.433, 0.380, 0.433, 0.445],
    'R2': [0.822, 0.843, 0.835, 0.887, 0.613, 0.667, 0.668, 0.630],
    'MSE': [0.140, 0.128, 0.129, 0.129, 0.307, 0.263, 0.263, 0.292],
    'MAE': [0.259, 0.245, 0.254, 0.254, 0.381, 0.338, 0.351, 0.360],
    'RMSE': [0.374, 0.357, 0.359, 0.360, 0.554, 0.513, 0.513, 0.541]
}

rsv_df = pd.DataFrame(rsv_data)

# 绘制RSV数据的图表
fig, axes = plt.subplots(5, 1, figsize=(10, 20))

metrics = ['Mape', 'R2', 'MSE', 'MAE', 'RMSE']
for ax, metric in zip(axes, metrics):
    for output_size in rsv_df['output_size'].unique():
        subset = rsv_df[rsv_df['output_size'] == output_size]
        ax.plot(subset['model'], subset[metric], marker='o', label=f'Output Size={output_size}')
    ax.set_ylabel(metric)
    ax.legend()
    ax.set_title(f'RSV Performance: {metric}')

plt.tight_layout()
plt.savefig('rsv.png')

