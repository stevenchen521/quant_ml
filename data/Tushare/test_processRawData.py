from unittest import TestCase
import tushare as ts
from functools import reduce
from data.Tushare.config import token
from data.Tushare.process_data import ProcessRawData
import pandas as pd


class TestProcessRawData(TestCase):
    def setUp(self):
        ts.set_token(token=token)

    def test_process_ticker_data(self):
        pro = ts.pro_api()
        df = pro.daily(ts_code='600276.SH', start_date='20010101', end_date='20190227')
        ProcessRawData.process_data_from_tushare(df=df,
                                                 save_code='600276SH',
                                                 col_need=['date', 'open', 'high', 'low', 'close', 'vol'],
                                                 rename={'vol': 'volume'})

    def test_process_shibor_data(self):
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

    def test_process_LPR_data(self):
        pro = ts.pro_api()
        df1 = pro.shibor_lpr(start_date='20010101', end_date='20100101')
        df2 = pro.shibor_lpr(start_date='20100102', end_date='20190307')

        df1 = ProcessRawData.process_data_from_tushare(df=df1,
                                                       col_need=['date', '1y'],
                                                       need_return=True)
        df2 = ProcessRawData.process_data_from_tushare(df=df2,
                                                       col_need=['date', '1y'],
                                                       need_return=True)
        df1 = df1.append(df2)
        df1.to_csv("..\..\data\shibor_lpr.csv")

    def test_process_account_data(self):
        pro = ts.pro_api()
        df1 = pro.stk_account(start_date='20010101', end_date='20100101')
        df2 = pro.stk_account(start_date='20100102', end_date='20190307')

        df1 = ProcessRawData.process_data_from_tushare(df=df1,
                                                       need_return=True)
        df2 = ProcessRawData.process_data_from_tushare(df=df2,
                                                       need_return=True)
        df1 = df1.append(df2)
        df1.to_csv("..\..\data\Account_number.csv")

    def test_merge_data(self):
        df1 = pd.read_csv('../../data/SH_index.csv')
        df2 = pd.read_csv('../../data/SH_index_growth.csv')
        df2.rename(columns={'Date': 'date'}, inplace=True)
        df2['date'] = df2['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        df2.sort_values(by=['date'], ascending=True, inplace=True)
        df3 = pd.read_csv('../../data/shibor.csv')
        df4 = pd.read_csv('../../data/shibor_lpr.csv')
        # df_temp = pd.merge(df1, df2, left_index=True, right_index=True, left_on='date', how='left')
        dfs = [df1, df2, df3, df4]
        df_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dfs)
        df_final = df_final.drop(['EOD_RISK_FREE_RATE_MID', 'FAIR_VALUE', 'DVD_SH_LAST', 'lpr_1y'], axis=1)
        df_final = df_final.dropna(how='any')
        df_final.index = df_final['date']
        del df_final['date']
        df_final.to_csv('../../data/SH_index_all.csv')
        # ProcessRawData.process_data_from_BBG(df1=df1, df2=df2, save_code='SH_index', add_col='IS_EPS')

    def test_merge_data1(self):
        df1 = pd.read_csv('../../data/SH_index.csv')
        df2 = pd.read_csv('../../data/SH_index_growth.csv')
        df2.rename(columns={'Date': 'date'}, inplace=True)
        df2['date'] = df2['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        df2.sort_values(by=['date'], ascending=True, inplace=True)
        df3 = pd.read_csv('../../data/shibor.csv')
        df4 = pd.read_csv('../../data/Tp_df.csv')
        df4['date'] = df4['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        # df_temp = pd.merge(df1, df2, left_index=True, right_index=True, left_on='date', how='left')
        dfs = [df1, df2, df3, df4]
        df_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dfs)
        # df_final = df_final.drop(['EOD_RISK_FREE_RATE_MID', 'FAIR_VALUE', 'DVD_SH_LAST'], axis=1)
        df_final.index = df_final['date']
        del df_final['date']
        # df_final = df_final[['open', 'high', 'low', 'close', 'volume', 'on', '1m',
        #                      '6m', '1y', 'IS_EPS', 'BEST_EPS', 'RETURN_COM_EQY', 'tp_score']]
        df_final = df_final[['open', 'high', 'low', 'close', 'volume', 'on', '6m', 'tp_score']]
        df_final = df_final.dropna(how='any')
        df_final.to_csv('../../data/SH_index_all.csv')


    def test_merge_data_add_nasdaq(self):
        df1 = pd.read_csv('../../data/SH_index.csv')
        df2 = pd.read_csv('../../data/SH_index_growth.csv')
        df2.rename(columns={'Date': 'date'}, inplace=True)
        df2['date'] = df2['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        df2.sort_values(by=['date'], ascending=True, inplace=True)

        df3 = pd.read_csv('../../data/shibor.csv')
        df4 = pd.read_csv('../../data/Tp_df.csv')
        df4['date'] = df4['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))

        df5 = pd.read_csv('../../data/nasdaq.csv')
        df5 = df5[['date', 'close', 'volume']].rename(columns={'close': 'nasclose', 'volume': 'nasvolume'})
        df5['date'] = df5['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        # df_temp = pd.merge(df1, df2, left_index=True, right_index=True, left_on='date', how='left')
        dfs = [df1, df2, df3, df4, df5]

        df_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='left'), dfs)
        # df_final = df_final.drop(['EOD_RISK_FREE_RATE_MID', 'FAIR_VALUE', 'DVD_SH_LAST'], axis=1)
        df_final.index = df_final['date']
        del df_final['date']
        # df_final = df_final[['open', 'high', 'low', 'close', 'volume', 'on', '1m',
        #                      '6m', '1y', 'IS_EPS', 'BEST_EPS', 'RETURN_COM_EQY', 'tp_score']]
        df_final = df_final[['open', 'high', 'low', 'close', 'volume', 'on', '6m', 'nasclose', 'nasvolume', 'tp_score']]
        df_final[['nasclose', 'nasvolume']] = df_final[['nasclose', 'nasvolume']].shift(-1)
        df_final[['nasclose', 'nasvolume']] = df_final[['nasclose', 'nasvolume']].fillna(method='backfill')
        df_final = df_final.dropna(how='any')
        df_final.to_csv('../../data/SH_index_all.csv')