# ####### <1>, get the final forecasting result for train and test period
# # 1.1 factor: Weather
WEATHER_COLUMNS="temp.max temp.min relative.humidity total.rainfall solar.radiation monthid weekid rate"
# ## 1.1.1 predict: ILI
python point/Baseline.py --data data/weather_data --predict_index ILI
python point/GRU.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
python point/GRUFcn.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
python point/LSTM.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
python point/LSTMFcn.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
python point/Seq2Seq.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
python point/TSTPlus.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
python point/InTimePlus.py --data data/weather_data --predict_index ILI --columns $WEATHER_COLUMNS
## 1.1.2 predict: RSV
python point/Baseline.py --data data/weather_data --predict_index RSV
python point/GRU.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS
python point/GRUFcn.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS
python point/LSTM.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS
python point/LSTMFcn.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS
python point/Seq2Seq.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS
python point/TSTPlus.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS
python point/InTimePlus.py --data data/weather_data --predict_index RSV --columns $WEATHER_COLUMNS

# 1.2 factor: h1_pos	sh3_pos	b_pos	Adenovirus monthid weekid rate
#             RSV_org	Adenovirus monthid weekid rate
ILI_COLUMNS="h1_pos	sh3_pos	b_pos	Adenovirus monthid weekid rate"
RSV_COLUMNS="RSV_org	Adenovirus monthid weekid rate"
## 1.1.1 predict: ILI
python point/Baseline.py --data data/pos_data --predict_index ILI
python point/GRU.py --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
python point/GRUFcn --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
python point/LSTM.py --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
python point/LSTMFcn.py --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
python point/Seq2Seq.py --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
python point/TSTPlus.py --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
python point/InTimePlus.py --data data/pos_data --predict_index ILI --columns $ILI_COLUMNS
## 1.1.2 predict: RSV
python point/Baseline.py ---data data/pos_data -predict_index RSV
python point/GRU.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS
python point/GRUFcn.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS
python point/LSTM.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS
python point/LSTMFcn.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS
python point/Seq2Seq.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS
python point/TSTPlus.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS
python point/InTimePlus.py --data data/pos_data --predict_index RSV --columns $RSV_COLUMNS