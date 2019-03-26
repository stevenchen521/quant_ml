import tushare as ts
from data.Tushare.config import token
import pandas as pd
from functools import reduce


class ProcessRawData(object):
    '''
    data download from tushare
    save_code: default false not to save; name of file to save.
    col_need: col need to save
    rename: default false no rename, else input a rename dictionary
    need_return: default false, else will return a processed dataframe
    '''
    @staticmethod
    def process_data_from_tushare(df, save_code=False, col_need=False, rename=False, need_return=False):
        if rename:
            df.rename(columns=rename, inplace=True)
        else:
            pass
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        df.sort_values(by=['date'], ascending=True, inplace=True)
        if col_need:
            df = df[col_need]
        else:
            pass
        df.index = df['date']
        del df['date']
        if save_code:
            df.to_csv("..\..\data\{}.csv".format(save_code))
        else:
            pass
        if need_return:
            print('process is finished')
            return df
        else:
            print('process is finished')

    def process_shibor_data(self):
        pro = ts.pro_api()
        df1 = pro.shibor(start_date='20120101', end_date='20190308')
        df2 = pro.shibor(start_date='20060101', end_date='20111231')
        df3 = pro.shibor(start_date='20010101', end_date='20051231')
        df1 = ProcessRawData.process_data_from_tushare(df=df1,
                                                       col_need=['date', 'on', '1m', '6m', '1y'],
                                                       need_return=True)
        df2 = ProcessRawData.process_data_from_tushare(df=df2,
                                                       col_need=['date', 'on', '1m', '6m', '1y'],
                                                       need_return=True)
        df3 = ProcessRawData.process_data_from_tushare(df=df3,
                                                       col_need=['date', 'on', '1m', '6m', '1y'],
                                                       need_return=True)
        df3 = df3.append(df2)
        df3 = df3.append(df1)
        df3.to_csv("..\..\data\shibor.csv")


    def merge_data1(self):
        df1 = pd.read_csv('../../data/SH_index.csv')
        df2 = pd.read_csv('../../data/SH_index_growth.csv')
        df2.rename(columns={'Date': 'date'}, inplace=True)
        df2['date'] = df2['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        df2.sort_values(by=['date'], ascending=True, inplace=True)
        df3 = pd.read_csv('../../data/shibor.csv')
        # df_temp = pd.merge(df1, df2, left_index=True, right_index=True, left_on='date', how='left')
        dfs = [df1, df2, df3]
        df_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dfs)
        # df_final = df_final.drop(['EOD_RISK_FREE_RATE_MID', 'FAIR_VALUE', 'DVD_SH_LAST'], axis=1)
        df_final.index = df_final['date']
        del df_final['date']
        df_final = df_final[['open', 'high', 'low', 'close', 'volume', 'on', '1m', '6m', '1y', 'IS_EPS', 'BEST_EPS', 'RETURN_COM_EQY']]
        df_final = df_final.dropna(how='any')
        df_final.to_csv('../../data/SH_index_all.csv')

if __name__ == '__main__':
    P = ProcessRawData()
    # P.process_shibor_data()
    P.merge_data1()


