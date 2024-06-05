# 1.1 Get ground truth curve
python EpidemicPeriod.py

# 1.2 Get the period result of testing models
python PeriodResult.py

# 1.3 Evaluate the model performance
python Evaluation.py --model GRU
python Evaluation.py --model GRUFcn
python Evaluation.py --model LSTM
python Evaluation.py --model LSTMFcn
python Evaluation.py --model Seq2Seq
python Evaluation.py --model InTimePlus
python Evaluation.py --model TSTPlus

# 1.4 Get accuracy result plot
python plot.py