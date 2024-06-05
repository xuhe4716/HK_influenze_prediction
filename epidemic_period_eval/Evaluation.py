import pandas as pd
import os
import csv
import argparse

class Evaluation:
    def __init__(self,ground_truth, evaluate_data):
        self.ground_truth = ground_truth
        self.evaluate_data = evaluate_data
        self.gt_df = pd.read_csv(self.ground_truth)
        self.evaluate_df = pd.read_csv(self.evaluate_data)


    def cal_epidemic_date(self, gt = True):
        if gt is True:
            df = self.gt_df
        else:
            df = self.evaluate_df
        def weekly_dates(row):
            start = pd.to_datetime(row['start_date'])
            end = pd.to_datetime(row['end_date'])
            return pd.date_range(start=start, end=end, freq='W-SUN').strftime('%Y-%m-%d').tolist()

        # Apply the function to each row in the DataFrame
        df['weekly_dates'] = df.apply(weekly_dates, axis=1)

        # Now create the nested dictionary
        result_dict = {} # Initialize an internal dictionary for each year with an empty list of winter and summer months
        for year in range(2009, 2020):
            result_dict[year] = {'winter': [], 'summer': [], "winter_peak":[], "summer_peak" : []}

        for index, row in df.iterrows():
            year = row['year_period']
            season = row['season']
            result_dict[year][season].extend(row['weekly_dates'])
            if season == "winter":
                result_dict[year]['winter_peak'].append((row['peak_date'],row['peak_rate']))
            elif season == "summer":
                result_dict[year]['summer_peak'].append((row['peak_date'],row['peak_rate']))

        return result_dict

    def epidemic_date_eval_confusion_matrix(self):
        # Evaluate the epidemic period
        gt_result_dict = self.cal_epidemic_date()
        model_result_dict = self.cal_epidemic_date(gt = False)

        confusion_matrix = {"tp":0,
                            "fp":0,
                            "fn":0,
                            "tn":0}
        total_date = 0

        for year_key, season_value in gt_result_dict.items():
            # get all date
            start_date = f'{year_key}-11-01'
            end_date = f'{year_key+1}-10-31'
            date_range = list(pd.date_range(start=start_date, end=end_date, freq='W'))
            date_list = [date.strftime('%Y-%m-%d') for date in date_range]

            gt_date = season_value['winter'] + season_value['summer']
            model_date = model_result_dict[year_key]['winter'] + model_result_dict[year_key]['summer']

            tp = list(set(gt_date).intersection(set(model_date))) # date with epidemic and predicted as an epidemic date
            fn = list(set(gt_date).difference(set(model_date))) # date with epidemic but not predicted as an epidemic date
            fp = list(set(model_date).difference(set(gt_date))) # date without epidemic but predicted as an epidemic date
            tn = list(set(date_list).difference(set(gt_date).union(set(model_date)))) # date without epidemic and not predicted as an epidemic date

            confusion_matrix['tp'] += len(tp)
            confusion_matrix['fn'] += len(fn)
            confusion_matrix['fp'] += len(fp)
            confusion_matrix['tn'] += len(tn)

            total_date += len(gt_date)

        sensitivity = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn']) # season with and predicted as an epidemic season
        specificity = confusion_matrix['tn'] / (confusion_matrix['tn'] + confusion_matrix['fp']) # epidemic with season but not predicted as an epidemic season
        precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp']) # epidemic without season but predicted as an epidemic season
        npv = confusion_matrix['tn'] / (confusion_matrix['tn'] + confusion_matrix['fn']) # epidemic without season and not predicted as an epidemic date
        accuracy = (confusion_matrix['tp'] + confusion_matrix['tn']) / (confusion_matrix['tn'] + confusion_matrix['fn'] + confusion_matrix['tp'] + confusion_matrix['fp'])
        f1 = (precision * sensitivity) / (precision + sensitivity)

        return sensitivity,specificity,precision,npv,accuracy

    def seasonal_eval_confusion_matrix(self):
        # Evaluate Whether Models Can Detect Seasonal Influenza
        gt_result_dict = self.cal_epidemic_date()
        model_result_dict = self.cal_epidemic_date(gt = False)

        confusion_matrix = {"tp":0,
                            "fp":0,
                            "fn":0,
                            "tn":0}

        for year_key, season_value in gt_result_dict.items():
            for season, date in season_value.items():
                if season == "winter" or season == "summer":
                    confusion_matrix['tp'] = confusion_matrix['tp'] + 1 if len(date) != 0 and len(model_result_dict[year_key][season]) != 0 else confusion_matrix['tp'] #
                    confusion_matrix['fn'] = confusion_matrix['fn'] + 1 if len(date) != 0 and len(model_result_dict[year_key][season]) == 0 else confusion_matrix['fn']
                    confusion_matrix['fp'] = confusion_matrix['fp'] + 1 if len(date) == 0 and len(model_result_dict[year_key][season]) != 0 else confusion_matrix['fp']
                    confusion_matrix['tn'] = confusion_matrix['tn'] + 1 if len(date) == 0 and len(model_result_dict[year_key][season]) == 0 else confusion_matrix['tn']


        sensitivity = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'])
        specificity = confusion_matrix['tn'] / (confusion_matrix['tn'] + confusion_matrix['fp'])
        precision = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'])
        npv = confusion_matrix['tn'] / (confusion_matrix['tn'] + confusion_matrix['fn'])
        accuracy = (confusion_matrix['tp'] + confusion_matrix['tn']) / (confusion_matrix['tn'] + confusion_matrix['fn'] + confusion_matrix['tp'] + confusion_matrix['fp'])
        f1 = (precision * sensitivity) / (precision + sensitivity)

        return sensitivity,specificity,precision,npv,accuracy


    def peak_date_eval_mae_mse(self):
        gt_result_dict = self.cal_epidemic_date()
        model_result_dict = self.cal_epidemic_date(gt = False)

        def weeks_difference(date_str1, date_str2):
            # Convert date strings to datetime objects
            date1 = pd.to_datetime(date_str1)
            date2 = pd.to_datetime(date_str2)
            # Calculate the difference in dates
            date_diff = abs(date2 - date1)
            # Convert the difference to weeks
            weeks_diff_mae = date_diff.days / 7
            weeks_diff_mse = weeks_diff_mae ** 2
            return weeks_diff_mae, weeks_diff_mse

        def peak_rate_difference(rate1, rate2):
            rate1 = float(rate1)
            rate2 = float(rate2)

            rate_diff_mae = abs(rate2 - rate1)
            rate_diff_mse = rate_diff_mae ** 2
            return rate_diff_mae,rate_diff_mse

        diff_weeks_sum_mae =0
        diff_weeks_sum_mse = 0
        diff_rate_sum_mae = 0
        diff_rate_sum_mse = 0
        effective_year = 0
        for year_key, season_value in gt_result_dict.items():
            for season_peak, value in season_value.items():
                if season_peak == "winter_peak" or season_peak == "summer_peak":
                    if len(value) != 0:
                        gt_peak_date,gt_peak_value = max(value, key=lambda x: x[1])
                    else:
                        gt_peak_date,gt_peak_value = (None, 0)
                    if len(model_result_dict[year_key][season_peak]) != 0:
                        model_peak_date,model_peak_value = max(model_result_dict[year_key][season_peak], key=lambda x: x[1])
                    else:
                        model_peak_date,model_peak_value = (None, 0)

                    if gt_peak_date is not None and model_peak_date is not None:
                        weeks_difference_mae, weeks_difference_mse= weeks_difference(gt_peak_date,model_peak_date)
                        peak_rate_difference_mae, peak_rate_difference_mse = peak_rate_difference(gt_peak_value,model_peak_value)

                        diff_weeks_sum_mae += weeks_difference_mae
                        diff_weeks_sum_mse += weeks_difference_mse
                        diff_rate_sum_mae += peak_rate_difference_mae
                        diff_rate_sum_mse += peak_rate_difference_mse
                        effective_year += 1
                    elif gt_peak_date is None and model_peak_date is None:
                        effective_year += 1

        week_diff_mae = diff_weeks_sum_mae / effective_year
        rate_diff_mae = diff_rate_sum_mae / effective_year
        week_diff_mse = diff_weeks_sum_mse / effective_year
        rate_diff_mse = diff_rate_sum_mae / effective_year

        return week_diff_mae,rate_diff_mae,week_diff_mse,rate_diff_mse




    def peak_date_eval_confusion_matrix(self,mode = 'strict'):
        # evaluate the peak date and peak rate with confusion_matrix
        # strict mode: peak date exactly matched observations and peak rate fell within ±20% of the observed
        # loose mode: peak date within ±2 weeks of observation and peak rate fell within ±50% of the observed

        gt_result_dict = self.cal_epidemic_date()
        model_result_dict = self.cal_epidemic_date(gt = False)

        def weeks_difference(date_str1, date_str2):
            # Convert date strings to datetime objects
            date1 = pd.to_datetime(date_str1)
            date2 = pd.to_datetime(date_str2)
            # Calculate the difference in dates
            date_diff = abs(date2 - date1)
            # Convert the difference to weeks
            weeks_diff = date_diff.days / 7
            #weeks_diff_mse = weeks_diff_mae ** 2
            return weeks_diff

        def peak_rate_difference(rate1, rate2):
            rate1 = float(rate1)
            rate2 = float(rate2)

            rate_diff_mae = abs(rate2 - rate1)
            #rate_diff_mse = rate_diff_mae ** 2
            return rate_diff_mae

        confusion_matrix_peak_week_diff = {"tp":0,
                            "fp":0,
                            "fn":0,
                            "tn":0}

        confusion_matrix_peak_rate_diff = {"tp":0,
                                           "fp":0,
                                           "fn":0,
                                           "tn":0}

        confusion_matrix_week_and_rate = {"tp":0,
                                          "fp":0,
                                          "fn":0,
                                          "tn":0}
        total_date = 0
        for year_key, season_value in gt_result_dict.items():
            for season_peak, value in season_value.items():
                if season_peak == "winter_peak" or season_peak == "summer_peak":
                    if len(value) != 0:
                        gt_peak_date,gt_peak_value = max(value, key=lambda x: x[1])
                    else:
                        gt_peak_date,gt_peak_value = (None, 0)
                    if len(model_result_dict[year_key][season_peak]) != 0:
                        model_peak_date,model_peak_value = max(model_result_dict[year_key][season_peak], key=lambda x: x[1])
                    else:
                        model_peak_date,model_peak_value = (None, 0)

                    if gt_peak_date is not None and model_peak_date is not None:
                        week_diff= weeks_difference(gt_peak_date,model_peak_date)
                        rate_diff = peak_rate_difference(gt_peak_value,model_peak_value)

                        if mode == "strict":
                            if week_diff == 0:
                                confusion_matrix_peak_week_diff['tp'] += 1
                            else:
                                confusion_matrix_peak_week_diff['fn'] += 1
                            if model_peak_value <= (gt_peak_value + gt_peak_value * 0.2) and model_peak_value >= (gt_peak_value - gt_peak_value * 0.2):
                                confusion_matrix_peak_rate_diff['tp'] += 1
                            else:
                                confusion_matrix_peak_rate_diff['fn'] += 1

                        elif mode == "loose":
                            if week_diff <= 2:
                                confusion_matrix_peak_week_diff['tp'] += 1
                            else:
                                confusion_matrix_peak_week_diff['fn'] += 1

                            if model_peak_value <= (gt_peak_value + gt_peak_value * 0.5) and model_peak_value >= (gt_peak_value - gt_peak_value * 0.5):
                                confusion_matrix_peak_rate_diff['tp'] += 1
                            else:
                                confusion_matrix_peak_rate_diff['fn'] += 1

                    confusion_matrix_peak_week_diff['fn'] = confusion_matrix_peak_week_diff['fn'] + 1 if gt_peak_date is not None and model_peak_date is None else confusion_matrix_peak_week_diff['fn']
                    confusion_matrix_peak_rate_diff['fn'] = confusion_matrix_peak_rate_diff['fn'] + 1 if gt_peak_value is not None and model_peak_value is None else confusion_matrix_peak_rate_diff['fn']
                    confusion_matrix_peak_week_diff['tn'] = confusion_matrix_peak_week_diff['tn'] + 1 if gt_peak_date is None and model_peak_date is None else confusion_matrix_peak_week_diff['tn']
                    confusion_matrix_peak_rate_diff['tn'] = confusion_matrix_peak_rate_diff['tn'] + 1 if gt_peak_value is None and model_peak_value is None else confusion_matrix_peak_rate_diff['tn']
                    confusion_matrix_peak_week_diff['fp'] = confusion_matrix_peak_week_diff['fp'] + 1 if gt_peak_date is None and model_peak_date is not None else confusion_matrix_peak_week_diff['fp']
                    confusion_matrix_peak_rate_diff['fp'] = confusion_matrix_peak_rate_diff['fp'] + 1 if gt_peak_value is None and model_peak_value is not None else confusion_matrix_peak_rate_diff['fp']

        #peak_week_sensitivity = confusion_matrix_peak_week_diff['tp'] / (confusion_matrix_peak_week_diff['tp'] + confusion_matrix_peak_week_diff['fn'])
        #peak_week_specificity = confusion_matrix_peak_week_diff['tn'] / (confusion_matrix_peak_week_diff['tn'] + confusion_matrix_peak_week_diff['fp'])
        #peak_week_precision = confusion_matrix_peak_week_diff['tp'] / (confusion_matrix_peak_week_diff['tp'] + confusion_matrix_peak_week_diff['fp'])
        #peak_week_npv = confusion_matrix_peak_week_diff['tn'] / (confusion_matrix_peak_week_diff['tn'] + confusion_matrix_peak_week_diff['fn'])
        peak_week_accuracy = (confusion_matrix_peak_week_diff['tp'] + confusion_matrix_peak_week_diff['tn']) / (confusion_matrix_peak_week_diff['tn'] + confusion_matrix_peak_week_diff['fn'] + confusion_matrix_peak_week_diff['tp'] + confusion_matrix_peak_week_diff['fp'])

        #peak_rate_sensitivity = confusion_matrix_peak_rate_diff['tp'] / (confusion_matrix_peak_rate_diff['tp'] + confusion_matrix_peak_rate_diff['fn'])
        #peak_rate_specificity = confusion_matrix_peak_rate_diff['tn'] / (confusion_matrix_peak_rate_diff['tn'] + confusion_matrix_peak_rate_diff['fp'])
        #peak_rate_precision = confusion_matrix_peak_rate_diff['tp'] / (confusion_matrix_peak_rate_diff['tp'] + confusion_matrix_peak_rate_diff['fp'])
        #peak_rate_npv = confusion_matrix_peak_rate_diff['tn'] / (confusion_matrix_peak_rate_diff['tn'] + confusion_matrix_peak_rate_diff['fn'])
        peak_rate_accuracy = (confusion_matrix_peak_rate_diff['tp'] + confusion_matrix_peak_rate_diff['tn']) / (confusion_matrix_peak_rate_diff['tn'] + confusion_matrix_peak_rate_diff['fn'] + confusion_matrix_peak_rate_diff['tp'] + confusion_matrix_peak_rate_diff['fp'])


        return peak_week_accuracy,peak_rate_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Baseline', help='model name')
    args = parser.parse_args()
    model_name = args.model

    gt = "PeriodResult/ground_truth/result/ground_truth_0.csv"
    directory_path_eval_df = f"PeriodResult/model/{model_name}/result"
    metrics_result_folder = 'MetricsResult/'
    if not os.path.exists(metrics_result_folder):
        os.makedirs(metrics_result_folder)

    print(f"evaluating model {model_name}...")

    for i in range(len(os.listdir(directory_path_eval_df))):
        eval_df = f"{directory_path_eval_df}/{model_name}_{i}.csv"
        if os.path.isfile(eval_df):
            n_weeks_ahead = i
            E = Evaluation(gt,eval_df)
            sensitivity_date,specificity_date,precision_date,npv_date,accuray = E.epidemic_date_eval_confusion_matrix()
            sensitivity_seasonal,specificity_seasonal,precision_seasonal,npv_seasonal,accuracy_seasonal = E.seasonal_eval_confusion_matrix()
            #week_diff_mae,rate_diff_mae,week_diff_mse,rate_diff_mse = E.peak_date_eval()
            strict_peak_week_accuracy,strict_peak_rate_accuracy = E.peak_date_eval_confusion_matrix()
            loose_peak_week_accuracy,loose_peak_rate_accuracy = E.peak_date_eval_confusion_matrix(mode = "loose")

            date_info = [model_name, n_weeks_ahead, sensitivity_date,specificity_date,precision_date,npv_date,accuray]
            seasonal_info = [model_name, n_weeks_ahead, sensitivity_seasonal,specificity_seasonal,precision_seasonal,npv_seasonal,accuracy_seasonal]
            peak_diff_info = [model_name, n_weeks_ahead, strict_peak_week_accuracy,strict_peak_rate_accuracy,loose_peak_week_accuracy,loose_peak_rate_accuracy]

            # save the result
            with open(f'{metrics_result_folder}/epidemicPeriod_diff.csv', 'a', newline='') as period_f:
                writer = csv.writer(period_f)
                writer.writerow(date_info)

            with open(f'{metrics_result_folder}/epidemicSeasonal_diff.csv', 'a', newline='') as seasonal_f:
                writer = csv.writer(seasonal_f)
                writer.writerow(seasonal_info)

            with open(f'{metrics_result_folder}/epidemicPeak_diff.csv', 'a', newline='') as peak_f:
                writer = csv.writer(peak_f)
                writer.writerow(peak_diff_info)






