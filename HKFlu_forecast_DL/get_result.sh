####### <1>, get the final forecasting result for train and test period
# 1.1 factor: Weather
## 1.1.1 predict: ILI
python plot.py --data data/weather_data --predict_index ILI --model Baseline
python plot.py --data data/weather_data --predict_index ILI --model GRU
python plot.py --data data/weather_data --predict_index ILI --model GRUFcn
python plot.py --data data/weather_data --predict_index ILI --model LSTM
python plot.py --data data/weather_data --predict_index ILI --model LSTMFcn
python plot.py --data data/weather_data --predict_index ILI --model Seq2Seq
python plot.py --data data/weather_data --predict_index ILI --model TSTPlus
python plot.py --data data/weather_data --predict_index ILI --model InTimePlus
## 1.1.2 predict: RSV
python plot.py --data data/weather_data --predict_index RSV --model Baseline
python plot.py --data data/weather_data --predict_index RSV --model GRU
python plot.py --data data/weather_data --predict_index RSV --model GRUFcn
python plot.py --data data/weather_data --predict_index RSV --model LSTM
python plot.py --data data/weather_data --predict_index RSV --model LSTMFcn
python plot.py --data data/weather_data --predict_index RSV --model Seq2Seq
python plot.py --data data/weather_data --predict_index RSV --model TSTPlus
python plot.py --data data/weather_data --predict_index RSV --model InTimePlus

# 1.2 factor: h1_pos	sh3_pos	b_pos	Adenovirus monthid weekid rate
#             RSV_org	Adenovirus monthid weekid rate
## 1.1.1 predict: ILI
python plot.py --data data/pos_data --predict_index ILI --model Baseline
python plot.py --data data/pos_data --predict_index ILI --model GRU
python plot.py --data data/pos_data --predict_index ILI --model GRUFcn
python plot.py --data data/pos_data --predict_index ILI --model LSTM
python plot.py --data data/pos_data --predict_index ILI --model LSTMFcn
python plot.py --data data/pos_data --predict_index ILI --model Seq2Seq
python plot.py --data data/pos_data --predict_index ILI --model TSTPlus
python plot.py --data data/pos_data --predict_index ILI --model InTimePlus
## 1.1.2 predict: RSV
python plot.py --data data/pos_data --predict_index RSV --model Baseline
python plot.py --data data/pos_data --predict_index RSV --model GRU
python plot.py --data data/pos_data --predict_index RSV --model GRUFcn
python plot.py --data data/pos_data --predict_index RSV --model LSTM
python plot.py --data data/pos_data --predict_index RSV --model LSTMFcn
python plot.py --data data/pos_data --predict_index RSV --model Seq2Seq
python plot.py --data data/pos_data --predict_index RSV --model TSTPlus
python plot.py --data data/pos_data --predict_index RSV --model InTimePlus
