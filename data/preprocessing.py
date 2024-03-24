import pandas as pd


def dataframe_preprocessing(file,factor_virus = ('Adenovirus'),
                            predict_virus = ('h1_pos','sh3_pos','b_pos','RSV','paraflu12', 'paraflu34')):

    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    df['virus_pos'] = 0
    for pos in ['h1_pos','sh3_pos','b_pos']:
        if pos in predict_virus:
            df['virus_pos'] += df[pos]

    df_final = pd.DataFrame()
    df_final['date']  = df['date']
    df_final['pos_index'] = df['rate.All'] * df['virus_pos']

    for flu in ['Adenovirus', 'paraflu12', 'paraflu34', 'RSV']:
        if flu in predict_virus or flu in factor_virus:
            df_final[flu] = df['rate.All'] * df[flu]
            #df_final[flu] = df_final[flu].shift(1)

    weather_factor = ['temp.max','temp.mean','temp.min','relative.humidity',
                      'total.rainfall','solar.radiation','wind.speed',
                      'absolute.humidity','pressure','temp.range']

    for factor in weather_factor:
        df_final[factor] = df[factor]

    df_final = df_final.reset_index(drop = True)
    return df_final

def main():
    df = dataframe_preprocessing('data_org.csv')
    df.to_csv("data_pp.csv",index=False)
if __name__ == "__main__":
    main()