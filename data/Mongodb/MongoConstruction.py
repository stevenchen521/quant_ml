import pymongo
import tushare as ts
from data.Tushare.config import token
from data.Tushare.process_data import ProcessRawData
import datetime
import json
import time
import pandas as pd


class MongoBase(object):
    def __init__(self, db, collection, local=True, user=None, password=None, host=None, port =None,auth_db = None):
        self.db = db
        self.collection = collection
        self.local = local
        self.user = user
        self.passwd = password
        self.host = host
        self.port = port
        self.authdb = auth_db
        self.OpenDB()

    def OpenDB(self):
        if self.local:
            self.con = pymongo.MongoClient("mongodb://localhost:27017/")
        else:
            uri = "mongodb://"+self.user+":"+self.passwd+"@"+self.host+":"+self.port+"/"+self.authdb+"?authMechanism=SCRAM-SHA-1"
            self.con = pymongo.MongoClient(uri, connect=False)
        self.db = self.con[self.db]
        self.collection = self.db[self.collection]

    def closeDB(self):
        self.con.close()



class TushareMongo(object):
    def __init__(self):
        ts.set_token(token=token)
        self.today = datetime.datetime.now().strftime('20%y%m%d')


    def get_ts_code(self):
        pro = ts.pro_api()
        df = pro.query('stock_basic', exchange='', list_status='L',
                         fields='ts_code,symbol,name,area,industry,list_date')
        mongo = MongoBase(db='ChinaStockData', collection='ts_code')
        mongo.collection.insert_many(json.loads(df.to_json(orient='records')))
        # print(x.inserted_ids)
        mongo.closeDB()



    def download_daily_price(self):
        pro = ts.pro_api()
        mongo = MongoBase(db='ChinaStockData', collection='ts_code')
        results = mongo.collection.find({}, {'ts_code': 1, '_id': 0})
        code_df = pd.DataFrame(list(results))
        print(len(code_df))
        for index, row in code_df.iterrows():
            ts_code = row['ts_code']
            mongo = MongoBase(db='ChinaStockData', collection=ts_code)
            df = pro.query('daily', ts_code=ts_code, start_date='20000101', end_date=self.today)
            mongo.collection.drop()
            mongo.collection.insert_many(json.loads(df.to_json(orient='records')))
            mongo.closeDB()
            time.sleep(0.3)
            print(index)


    # def download_sector_index(self):








    def main(self):
        # self.get_ts_code()
        self.download_daily_price()


if __name__ == '__main__':
    T = TushareMongo()
    T.main()










