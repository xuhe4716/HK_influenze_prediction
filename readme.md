# Hong Kong Epidemic Forecast

## 1.Description

This repository provides methods for predicting Hong Kong epidemics. [HKFlu_forecast_DL](HKFlu_forecast_DL) trains a variety of deep learning models to predict Hong Kong epidemics, and [epidemic_period_eval](epidemic_period_eval)  uses the results obtained from [HKFlu_forecast_DL](HKFlu_forecast_DL) for evaluating epidemic intervals and peaks.

```
├── HK_influenze_prediction
│   ├── HKFlu_forecast_DL
│   ├── epidemic_period_eval
│   └── readme.md
```



## 2.Rerun the code - Deep Learning Forestcast

## 2.1. Make sure the right environment

First, install python and conda from https://www.anaconda.com/download; install Rtools from https://cran.r-project.org/bin/windows/base/ or https://cran.r-project.org/bin/macosx/.

### Create torch environment

```bash
conda create -n torch_py39 python=3.9.17
conda activate torch_py39
conda install pytorch==2.2.0
conda install pandas==2.2.1
conda install matplotlib==3.8.4
conda install ipykernel==6.28.0
pip install tsai==0.3.7
pip install joblib==1.2.0
pip install --force-reinstall -v "optuna==3.2.0"
```

We also highly recommend to download the torch package following the command guidance in https://pytorch.org/.

<font size=4>**If you wish to reproduce the results without any bias, we strongly recommend running the provided code in a MacOs environment.**</font>

## 2.2 Rerun models

You can run the `bash run.sh` command to rerun the models and get all the result and figures.



## 3. Rerun the Code - Epidemic Period Evaluation

### 3.1 Required package
```angular2html
conda deactivate
pip install pandas==2.2.1
pip install matplotlib==3.8.4
pip install seaborn==0.11.0
```

### 3.2 Data

Before running the program, you need to run HKFlu_forecast_DL and get the forecast results from HKFlu_forecast_DL/Result/Point and copy the results to data/result in this folder. For example, we used HKFlu_forecast_DL/Scripts/data/weather_data/ILI.csv for model training, then in data/result are the results based on this dataset.

### 3.3 Get Evaluation result

You can run the `bash run.sh` command to rerun the models and get all the result and figures.

### 3.4 Definitions
#### How to identify an epidemic period?
**Start week:** If the rate is greater than 5 for two consecutive weeks, and the previous week was not labeled as an epidemic start week, then this week will be defined as the start week of the epidemic period. <br>
**End week:** If the rate is not greater than 5 for two consecutive weeks, then this week will be labeled as the possible end week. For each epidemic cycle, the date after the start date when the end condition is first met. If this date is later than the start date of the next period, the end date will be adjusted to the week before the start date of the next period.<br>
**Duration:** the number of weeks a period has lasted<br>
**Peak rate and peak date:** the maximum value of the rate in a period and the date of the week.<br>

#### What are the metrics for the evaluation?
We evaluated the performance of different models in recognizing epidemic period, epidemic season, and peak date and peak value, and used the confusion matrix as an evaluation metric. For each aspect of the evaluation, the definition of the confusion matrix will be slightly different. The general definitions of confusion matrix, sensitivity,specificity,precision,npv,accuracy are shown as below:<br>
|                              | Predicted as Positive| Predicted as Negative |
|------------------------------|--------------------------------|------------------------------------|
| Actual Positive       | True Positive (TP)             | False Negative (FN)                |
|Actual Negative     | False Positive (FP)            | True Negative (TN)                 |

$$
Sensitivity = \frac{TP}{TP + FN}
$$

$$
Specificity = \frac{TN}{TN + FP}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
NPV = \frac{TN}{TN + FN}
$$

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$



##### Epidemic season

From November 1 to October 31 of the following year is the epidemic period, determine whether the epidemic season generated by the model is consistent with the epidemic season in the ground truth. For example, if epidemics are detected in the winter of a given year, then at least one of the epidemics generated by the model is in the winter.

|                           | Seasons predicted to be with epidemic | Seasons predicted to be without epidemic |
| ------------------------- | ------------------------------------- | ---------------------------------------- |
| Seasons with epidemics    | True Positive (TP)                    | False Negative (FN)                      |
| Seasons without epidemics | False Positive (FP)                   | True Negative (TN)                       |

##### Epidemic period

A period from November 1 to October 31 of the following year was used to compare the epidemic period generated by the model with the ground truth.<br>
|                                  | Dates predicted to be with epidemic | Dates predicted to be without epidemic |
| -------------------------------- | ----------------------------------- | -------------------------------------- |
| Dates that with epidemics        | True Positive (TP)                  | False Negative (FN)                    |
| Dates that are without epidemics | False Positive (FP)                 | True Negative (TN)                     |

##### Peak date and peak rate

The peak date and peak rate in the epidemic PERIOD generated by each model were compared to the peak date and peak rate in the ground truth. Here we enforced two criteria, for the more stringent standard we required the peak date to match the observation exactly and the peak rate to be within +- 20% of the observation. For the looser standard, we allow a +- 2 week error between the peak date and the observation, and the peak rate to be within +- 50% of the observation.

|                                  | Dates predicted to be with epidemic/ Fulfill  the standard | Dates predicted to be without epidemic / Not Fulfill  the standard |
| -------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| Dates that with epidemics        | True Positive (TP)                                         | False Negative (FN)                                          |
| Dates that are without epidemics | False Positive (FP)                                        | True Negative (TN)                                           |



## \## 4.Authors 

**Xuefei He** - *Initial Work* - [xuhe4716](https://github.com/xuhe4716)

