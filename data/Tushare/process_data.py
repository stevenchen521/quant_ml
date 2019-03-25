import tushare as ts
from data.Tushare.config import token
import pandas as pd

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


















