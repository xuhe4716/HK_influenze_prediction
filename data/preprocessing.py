import pandas as pd


def dataframe_preprocessing(file,factor_virus = ('Adenovirus'),
                            predict_virus = ('h1_pos','sh3_pos','b_pos','RSV','paraflu12', 'paraflu34')):

    # date, week, month, year id preprocessing
    df = pd.read_csv(file)
    df_pos_index = pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df_pos_index['date']  = df['date']
    df_pos_index['monthid'] = df['monthid']
    df_pos_index[['yearid', 'weekid']] = df['uid'].str.split('-', expand=True)


    df['virus_pos'] = 0
    for pos in ['h1_pos','sh3_pos','b_pos']:
        df_pos_index[pos] = df[pos]
        if pos in predict_virus:
            df['virus_pos'] += df[pos]


    df_pos_index['pos_index'] = df['rate.All'] * df['virus_pos']
    df_pos_index['RSV_org'] = df['RSV']

    for flu in ['Adenovirus', 'paraflu12', 'paraflu34', 'RSV']:
        if flu in predict_virus or flu in factor_virus:
            df_pos_index[flu] = df['rate.All'] * df[flu]
            #df_final[flu] = df_final[flu].shift(1)



    weather_factor = ['temp.max','temp.mean','temp.min','relative.humidity',
                      'total.rainfall','solar.radiation','wind.speed',
                      'absolute.humidity','pressure','temp.range']

    for factor in weather_factor:
        df_pos_index[factor] = df[factor]

    df_final = df_pos_index.reset_index(drop = True)
    return df_final

def main():
    df = dataframe_preprocessing('data_org.csv')
    df.to_csv("data_pp.csv",index=False)

    df_pos = df.drop(columns =  ['temp.max','temp.mean','temp.min','relative.humidity',
                                 'total.rainfall','solar.radiation','wind.speed',
                                 'absolute.humidity','pressure','temp.range',"RSV",'paraflu12','paraflu34','RSV_org'])
    df_pos.to_csv('pos_index_prediction.csv',index=False)

    df_rsv = df.drop(columns =  ['temp.max','temp.mean','temp.min','relative.humidity',
                                 'total.rainfall','solar.radiation','wind.speed',
                                 'absolute.humidity','pressure','temp.range',"pos_index",'paraflu12','paraflu34','h1_pos','sh3_pos','b_pos'])
    df_rsv.to_csv('rsv_predction.csv',index=False)
if __name__ == "__main__":
    main()