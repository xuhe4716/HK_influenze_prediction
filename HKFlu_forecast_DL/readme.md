# Result
** The following shows the ratio of each model's metrics to the base model's **
| Baseline   | 0 | 2.19  | 0.38 | 0.15 | 1.3  | 1           | 1           | 1           | 1           |
|------------|---|-------|------|------|------|-------------|-------------|-------------|-------------|
| Baseline   | 1 | 3.61  | 0.66 | 0.22 | 2.08 | 1           | 1           | 1           | 1           |
| Baseline   | 4 | 7.75  | 1.74 | 0.37 | 4.2  | 1           | 1           | 1           | 1           |
| Baseline   | 8 | 7.95  | 2.03 | 0.47 | 5.48 | 1           | 1           | 1           | 1           |
| GRU        | 0 | 1.98  | 0.33 | 0.14 | 1.22 | 0.904109589 | 0.868421053 | 0.933333333 | 0.938461538 |
| GRU        | 1 | 3.51  | 0.76 | 0.26 | 2.3  | 0.972299169 | 1.151515152 | 1.181818182 | 1.105769231 |
| GRU        | 4 | 4.86  | 1.19 | 0.3  | 2.93 | 0.627096774 | 0.683908046 | 0.810810811 | 0.697619048 |
| GRU        | 8 | 7.4   | 1.19 | 0.36 | 4.37 | 0.93081761  | 0.586206897 | 0.765957447 | 0.797445255 |
| GRUFcn     | 0 | 2.03  | 0.42 | 0.17 | 1.26 | 0.926940639 | 1.105263158 | 1.133333333 | 0.969230769 |
| GRUFcn     | 1 | 3.47  | 0.67 | 0.26 | 2.21 | 0.961218837 | 1.015151515 | 1.181818182 | 1.0625      |
| GRUFcn     | 4 | 4.66  | 1.19 | 0.33 | 3    | 0.601290323 | 0.683908046 | 0.891891892 | 0.714285714 |
| GRUFcn     | 8 | 6.73  | 1.12 | 0.35 | 4.1  | 0.846540881 | 0.551724138 | 0.744680851 | 0.748175182 |
| LSTM       | 0 | 4.46  | 0.46 | 0.18 | 2.11 | 2.03652968  | 1.210526316 | 1.2         | 1.623076923 |
| LSTM       | 1 | 4.39  | 0.73 | 0.29 | 2.63 | 1.216066482 | 1.106060606 | 1.318181818 | 1.264423077 |
| LSTM       | 4 | 5.13  | 1.09 | 0.33 | 3.12 | 0.661935484 | 0.626436782 | 0.891891892 | 0.742857143 |
| LSTM       | 8 | 6.05  | 0.98 | 0.35 | 3.8  | 0.761006289 | 0.482758621 | 0.744680851 | 0.693430657 |
| LSTMFcn    | 0 | 1.93  | 0.28 | 0.12 | 1.14 | 0.881278539 | 0.736842105 | 0.8         | 0.876923077 |
| LSTMFcn    | 1 | 15.08 | 1.4  | 0.3  | 5.52 | 4.177285319 | 2.121212121 | 1.363636364 | 2.653846154 |
| LSTMFcn    | 4 | 7.11  | 1.29 | 0.37 | 4.02 | 0.917419355 | 0.74137931  | 1           | 0.957142857 |
| LSTMFcn    | 8 | 6.4   | 1.13 | 0.38 | 4.1  | 0.805031447 | 0.556650246 | 0.808510638 | 0.748175182 |
| Seq2Seq    | 0 | 3.13  | 0.37 | 0.16 | 1.62 | 1.429223744 | 0.973684211 | 1.066666667 | 1.246153846 |
| Seq2Seq    | 1 | 4.59  | 0.75 | 0.26 | 2.57 | 1.271468144 | 1.136363636 | 1.181818182 | 1.235576923 |
| Seq2Seq    | 4 | 6.67  | 1.4  | 0.36 | 3.81 | 0.860645161 | 0.804597701 | 0.972972973 | 0.907142857 |
| Seq2Seq    | 8 | 6.6   | 1.16 | 0.4  | 4.37 | 0.830188679 | 0.571428571 | 0.85106383  | 0.797445255 |
| TSTPlus    | 0 | 5.54  | 1.13 | 0.37 | 3.33 | 2.529680365 | 2.973684211 | 2.466666667 | 2.561538462 |
| TSTPlus    | 1 | 4.31  | 0.86 | 0.34 | 2.77 | 1.193905817 | 1.303030303 | 1.545454545 | 1.331730769 |
| TSTPlus    | 4 | 4.93  | 1.38 | 0.39 | 3.26 | 0.636129032 | 0.793103448 | 1.054054054 | 0.776190476 |
| TSTPlus    | 8 | 6.46  | 1.27 | 0.44 | 4.4  | 0.812578616 | 0.625615764 | 0.936170213 | 0.802919708 |
| InTimePlus | 0 | 21.94 | 1.05 | 0.31 | 4.61 | 10.01826484 | 2.763157895 | 2.066666667 | 3.546153846 |
| InTimePlus | 1 | 6.62  | 1.21 | 0.31 | 3.43 | 1.833795014 | 1.833333333 | 1.409090909 | 1.649038462 |
| InTimePlus | 4 | 5.66  | 1.46 | 0.39 | 3.73 | 0.730322581 | 0.83908046  | 1.054054054 | 0.888095238 |
| InTimePlus | 8 | 7.38  | 1.7  | 0.44 | 4.95 | 0.928301887 | 0.837438424 | 0.936170213 | 0.903284672 |