"""
Epidemic Period - Ground truth
Author: He
"""


import pandas as pd
import numpy as np
np.set_printoptions(suppress = True)
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os



class EpidemicPeriod:
    def __init__(self,data_org,start_date,end_date,nweek_ahead, model_type = "ground_truth", format = True):
        self.data_rt = None
        self.start_date = start_date
        self.end_date = end_date
        self.model_type = model_type
        self.nweek_ahead = nweek_ahead
        self.format = format
        self.daily = self.data_reader(data_org,start_date,end_date,format = self.format)
        self.result = self.epidemic_calculate()


    def data_reader(self,data_rt, start_date,end_date,format = True):
        #data_rt = pd.read_csv(filename)
        if format is True:
            data_rt['date'] = pd.to_datetime(data_rt['date'], format='%d/%m/%Y')
        else:
            data_rt['date'] = pd.to_datetime(data_rt['date'], format='%Y-%m-%d')

        # data filter and add columns
        data_rt = data_rt[data_rt['date'] >= start_date] # ?
        data_rt['year'] = data_rt['date'].dt.year
        data_rt['monthid'] = data_rt['date'].dt.month
        data_rt['season'] = 'summer'
        # set winter month
        winter_months = [11, 12, 1, 2, 3, 4]
        data_rt.loc[data_rt['monthid'].isin(winter_months), 'season'] = 'winter'
        month_ids = data_rt['monthid'].unique()
        data_rt['monthid'] = pd.Categorical(data_rt['monthid'], categories=month_ids, ordered=True)
        # set test set
        self.data_rt = data_rt

        def get_daily(self):
            # convert the data to daily data
            n = len(data_rt)
            temp = np.full((n * 7, 3), np.nan)
            temp[:, 0] = np.arange(1, int(n * 7 + 1))
            for i in range(1, len(temp) + 1):
                temp[i - 1][1] = data_rt['rate'][i / 7 - 1] if i % 7 == 0 else None
            temp[0][1] = temp[0][2] = 0
            temp[len(temp) - 1][1] = temp[len(temp) - 1][2] = 0
            # use linear smooth to fill the na
            temp_df = pd.DataFrame(temp, columns = ["index","rate","approx_rate"])
            temp_df['approx_rate'] = temp_df['rate'].interpolate(method='linear')

            daily_date = pd.date_range(start=start_date, end=end_date, freq='D')
            dat_daily = pd.DataFrame({
                'date': daily_date,
                'rate': temp_df['approx_rate'].iloc[:len(daily_date)]
            })

            dat_daily['date'] = pd.to_datetime(dat_daily['date'])
            dat_daily['year'] = dat_daily['date'].dt.strftime('%y')  # 提取年份的最后两位
            dat_daily['monthid'] = dat_daily['date'].dt.month  # 提取月份
            dat_daily['season'] = 'summer'
            winter_months = [11, 12, 1, 2, 3, 4]
            dat_daily.loc[dat_daily['monthid'].isin(winter_months), 'season'] = 'winter'

            return dat_daily

    def epidemic_calculate(self):
        # start
        dat_epidemic = self.data_rt.copy()
        dat_epidemic['today_more_than_5'] = 0
        dat_epidemic['tomorrow_more_than_5'] = 0
        dat_epidemic['could_start_today'] = 0
        dat_epidemic['could_start_yesterday'] = 0
        dat_epidemic['start'] = 0
        dat_epidemic['month_2_start'] = 0
        dat_epidemic['final_start'] = 0

        # 1. Compute 'today_more_than_5', mark as 1 if the rate exceeds 0.005
        dat_epidemic['today_more_than_5'] = (dat_epidemic['rate'] / 1000 > 0.005).astype(int)
        # 2. Shift the 'today_more_than_5' column values down by one row to 'tomorrow_more_than_5'
        dat_epidemic['tomorrow_more_than_5'] = dat_epidemic['today_more_than_5'].shift(-1).fillna(0).astype(int)
        # 3. Calculate 'could_start_today' where the sum of 'today_more_than_5' and 'tomorrow_more_than_5' equals 2
        dat_epidemic['could_start_today'] = ((dat_epidemic['today_more_than_5'] + dat_epidemic['tomorrow_more_than_5']) == 2).astype(int)
        # 4. Shift the 'could_start_today' column values down by one row to 'could_start_yesterday'
        dat_epidemic['could_start_yesterday'] = dat_epidemic['could_start_today'].shift(1).fillna(0).astype(int)
        # 5. Calculate 'start', marked as 1 when 'could_start_today' is 1 and 'could_start_yesterday' is 0
        dat_epidemic['start'] = ((dat_epidemic['could_start_today'] == 1) & (dat_epidemic['could_start_yesterday'] == 0)).astype(int)
        # 6. Directly copy the data from the 'start' column to 'month_2_start'
        dat_epidemic['month_2_start'] = dat_epidemic['start']

        for i in range(0, 8):
            ii = i + 1
            dat_epidemic['month_2_start'].iloc[ii:] += dat_epidemic['start'].iloc[:-(i + 1)].values

        condition = (dat_epidemic['start'] == 1) & (dat_epidemic['month_2_start'] <= 1)
        dat_epidemic.loc[condition, 'final_start'] = 1
        dat_epidemic.to_csv("dat_epidemic.csv",index = False)
        dat_start = dat_epidemic[['date', 'weekid', 'season', 'final_start']]
        dat_start['date'] = pd.to_datetime(dat_start['date'])

        # end
        dat_epidemic_end = self.data_rt.copy()
        dat_epidemic_end['today_more_than_5'] = 0
        dat_epidemic_end['tomorrow_more_than_5'] = 0
        dat_epidemic_end['could_end_today'] = 0
        dat_epidemic_end['could_end_yesterday'] = 0
        dat_epidemic_end['end'] = 0

        dat_epidemic_end.loc[dat_epidemic_end['rate'] / 1000 > 0.005, 'today_more_than_5'] = 1
        dat_epidemic_end['tomorrow_more_than_5'] = dat_epidemic_end['today_more_than_5'].shift(-1).fillna(0).astype(int)
        dat_epidemic_end['could_end_today'] = ((dat_epidemic_end['today_more_than_5'] + dat_epidemic_end['tomorrow_more_than_5']) == 0).astype(int)

        # duration
        dat_period = dat_start[dat_start['final_start'] == 1]
        dat_period = dat_period.rename(columns={'date': 'start_date'})
        dat_period['end_date'] = pd.to_datetime('1900-10-10')
        dat_period = dat_period.reset_index(drop = True)

        for i in range(len(dat_period)):
            start_date = dat_period.loc[i, 'start_date']
            filtered_end_dates = dat_epidemic_end[(dat_epidemic_end['could_end_today'] == 1) & (dat_epidemic_end['date'] > start_date)]
            end_date = filtered_end_dates['date'].min()

            if i < len(dat_period) - 1 and end_date >= dat_period['start_date'].loc[i + 1]:
                end_date = dat_period['start_date'].loc[i + 1] - pd.Timedelta(days=7)

            dat_period.loc[i, 'end_date'] = end_date

        dat_period['duration_weeks'] = ((dat_period['end_date'] - dat_period['start_date']).dt.days / 7).astype(int)

        # get peak
        dat_period['peak_rate'] = pd.NA
        dat_period['peak_date'] = pd.NaT
        dat_period['peak_weekid'] = 0

        dt_peak = self.data_rt.copy()
        dt_peak = dt_peak.set_index(dt_peak['date'])

        for index, row in dat_period.iterrows():
            start_date = row['start_date']
            end_date = row['end_date']

            # Filtering data by date range in dt_peak
            mask = (dt_peak.index >= start_date) & (dt_peak.index <= end_date)
            period_data = dt_peak.loc[mask]

            if not period_data.empty:
                # Find the maximum rate and its date
                peak_rate = period_data['rate'].max()
                peak_date = period_data['rate'].idxmax()
                peak_weekid = period_data.loc[peak_date, 'weekid']

                # Store results to dat_period
                dat_period.at[index, 'peak_rate'] = peak_rate
                dat_period.at[index, 'peak_date'] = peak_date
                dat_period.at[index, 'peak_weekid'] = peak_weekid

        # epidemic period
        dat_period['year_period'] = pd.NA
        dat_period['year_period'] = np.where(dat_period['start_date'].dt.month >= 11,
                                             dat_period['start_date'].dt.year,
                                             dat_period['start_date'].dt.year - 1)

        if self.model_type == "ground_truth":
            result_path = f"PeriodResult/{self.model_type}/result"
        else:
            result_path = f"PeriodResult/model/{self.model_type}/result"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        dat_period.to_csv(f"{result_path}/{self.model_type}_{self.nweek_ahead}.csv")

        return dat_period


    def get_plot(self):
        data_rt_plot= self.data_rt.copy()
        data_rt_plot = data_rt_plot.set_index(data_rt_plot['date'])
        plt.figure(figsize=(30, 12))
        dat_period = self.result

        plt.plot(data_rt_plot.index, data_rt_plot['rate'], color='black', label='Out of Period')

        for index, row in dat_period.iterrows():
            start_date = row['start_date']
            end_date = row['end_date']
            season = row['season']

            mask = (data_rt_plot.index >= start_date) & (data_rt_plot.index <= end_date)
            period_data = data_rt_plot.loc[mask]

            plt.plot(period_data.index, period_data['rate'],
                     color='blue' if season == 'winter' else 'red',
                     label=f'Period {index + 1} ({season})')

            plt.axvline(x=start_date, color='gray', linestyle='--', linewidth=2)
            plt.axvline(x=end_date, color='gray', linestyle='--', linewidth=2)

        #plt.legend(loc='best')
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xticks(rotation=45)

        plt.title('Rate Over Time by Season with Non-Period Rates')
        plt.xlabel('Date')
        plt.ylabel('Rate')

        if self.model_type == "ground_truth":
            result_path = f"PeriodResult/{self.model_type}/fig"
        else:
            result_path = f"PeriodResult/model/{self.model_type}/fig"
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        plt.savefig(f"{result_path}/{self.model_type}_{self.nweek_ahead}_Rate_Over_Time_Season.png")






if __name__ == "__main__":
    data_rt = pd.read_csv("data/ILI.csv")

    dataset = EpidemicPeriod(data_rt,start_date = "2009-11-01", end_date = "2019-12-29",nweek_ahead = 0)
    dataset.epidemic_calculate()
    dataset.get_plot()
