from unittest import TestCase
import pymongo
import tushare as ts
from data.Tushare.config import token
from data.Tushare.process_data import ProcessRawData
import datetime
import json
import pandas as pd
import shelve


class TestConnect(TestCase):
    def setUp(self):
        self.myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        ts.set_token(token=token)
        self.today = datetime.datetime.now().strftime('20%y%m%d')



    def test_insert_data(self):
        pro = ts.pro_api()
        df = pro.index_daily(ts_code='000001.SH', start_date='20010101', end_date=self.today)
        # ProcessRawData.process_data_from_tushare(df=df,
        #                                          save_code='SH_index',
        #                                          col_need=['date', 'open', 'high', 'low', 'close', 'volume'],
        #                                          rename={'vol': 'volume', 'trade_date': 'date'})
        mydb = self.myclient["ChinaStockData"]
        mycol = mydb['000001.SH']
        mycol.insert(json.loads(df.to_json(orient='records')))

        count = mycol.find().count()
        print(count)

    def test_read_Mongodb(self):
        db = self.myclient.ChinaStockData
        col = db['000001.SH']
        results = col.find().sort('trade_date', pymongo.ASCENDING)
        df = pd.DataFrame(list(results))
        del df['_id']
        print(df.head(5))
        print(type(df.trade_date.iloc[0]))




    # def test_mongodb_normal_process(self):


