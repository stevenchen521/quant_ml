import tushare as ts
from data.Tushare.config import token
import pandas as pd

def process_data(df, ts_code):
    # df.sort_values(by=['date'], ascending=True, inplace=True)
    df['date'] = df['trade_date'].apply(lambda x:pd.to_datetime(x).strftime("%Y-%m-%d"))
    df.sort_values(by=['date'], ascending=True, inplace=True)
    col_need = ['date', 'open', 'high', 'low', 'close', 'vol']
    df = df[col_need]
    df.rename(columns={'vol': 'volume'}, inplace=True)
    df.index = df['date']
    del df['date']
    df.to_csv("..\..\data\{}.csv".format(ts_code))
    print('process is finished')


class ProcessDownloadData(object):
    def __init__(self, df,  code_list, save_path='../data', default = True):
        self.df = df
        self.save_path = save_path
        self.default = default
        # self.code_list = code_list


if __name__ == '__main__':
    ts.set_token(token=token)
    pro = ts.pro_api()
    # df = pro.index_daily(ts_code='600276.SH', start_date='20010101', end_date='20190227')
    df = pro.daily(ts_code='600276.SH', start_date='20010101', end_date='20190227')
    process_data(df, '600276SH')



