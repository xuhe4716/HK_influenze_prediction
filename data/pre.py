import pandas as pd

df = pd.read_csv('pos_index_prediction.csv')
df_pos_index = pd.DataFrame()
df_pos_index['date'] = df['date']
df_pos_index['pos_index'] = df['pos_index']
df_pos_index.to_csv('pos_index.csv',index=False)

df_2 = pd.read_csv('rsv_predction.csv')
df_pos_index_2 = pd.DataFrame()
df_pos_index_2['date'] = df_2['date']
df_pos_index_2['pos_index'] = df_2['RSV']
df_pos_index_2.to_csv('RSV.csv',index=False)

df_pos_index_2['date'] = pd.to_datetime(df_pos_index_2['date'])
df_pos_index_2 = df_pos_index_2.set_index('date')
min_date = df_pos_index_2.index.min()
split_date = min_date + pd.DateOffset(weeks=52 * 12)
train_valid = df_pos_index_2[df_pos_index_2.index <= split_date]
test = df_pos_index_2[df_pos_index_2.index > split_date]
train_valid = train_valid.reset_index()
test = test.reset_index()
train_valid.to_csv('rsv_train.csv',index=False)
test.to_csv('rsv_test.csv',index=False)
