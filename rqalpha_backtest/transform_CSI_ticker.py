from os import listdir
import pandas as pd
# from os.path import isfile
dir = 'CSI300_DATA'
print(listdir(dir))

def transform_ticker_name(ticker):
    if ticker[7:9] == 'SZ':
        ticker_temp = ticker[0:6] + '.XSHE'
    else:
        ticker_temp = ticker[0:6] + '.XSHG'
    return ticker_temp

list_transformed = [transform_ticker_name(x) for x in listdir(dir)]
print(list_transformed)
df = pd.DataFrame({'ticker': pd.Series(list_transformed)}, index=range(len(list_transformed)))
df.to_csv('ticker_list.csv')
# for f in listdir(dir):

