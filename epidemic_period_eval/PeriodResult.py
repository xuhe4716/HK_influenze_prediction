"""
Epidemic Period - other models
Author: He
"""

import pandas as pd
from EpidemicPeriod import EpidemicPeriod
import os
class PeriodResult:
    def __init__(self):
        pass

    def data_reader(self,filename,n_week_ahead,model_type):
        # get period curve result of other model

        data_rt = pd.read_csv("data/ILI.csv")
        data_rt['date'] = pd.to_datetime(data_rt['date'], format='%d/%m/%Y')

        data_result = pd.read_csv(filename)
        data_result = data_result.loc[data_result['week_ahead'] == n_week_ahead]
        data_result = data_result[['date','point']]
        data_result['date'] = pd.to_datetime(data_result['date'], format='%Y-%m-%d')
        data_result.rename(columns={'point': 'rate'}, inplace=True)
        merged_data = pd.merge(data_result, data_rt[['date', 'weekid', 'monthid']], on='date', how='left')
        #print(merged_data)

        EP = EpidemicPeriod(merged_data,start_date = "2009-11-01", end_date = "2019-12-29",nweek_ahead = n_week_ahead, model_type = model_type,format = False)
        EP.epidemic_calculate()
        EP.get_plot()



if __name__ == "__main__":
    dataset = PeriodResult()
    directory_path = 'data/result'

    # Traverse through the files in the specified directory
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            model_type = filename.split("_")[1]
            print(f"generating period curve of model {model_type}......")

            #dataset.data_reader(f"{directory_path}/{filename}",5,model_type)
            for i in range(0,5):
                dataset.data_reader(f"{directory_path}/{filename}",i,model_type)

