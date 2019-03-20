import pymongo
import tushare as ts
from data.Tushare.config import token
from data.Tushare.process_data import ProcessRawData
import datetime
import json
import time
import pandas as pd
from data.Mongodb.MongoConstruction import MongoBase
import re
import os


class TushareHDF5(object):
    def __init__(self):
        ts.set_token(token=token)
        self.today = datetime.datetime.now().strftime('20%y%m%d')


    def ReadfromMongoDB(self):
        mongo = MongoBase(db='ChinaStockData', collection='ts_code')
        results = mongo.collection.find({}, {'ts_code': 1, '_id': 0})
        code_df = pd.DataFrame(list(results))
        project_dir = os.getcwd()
        mypath = re.findall(r'.*quant_ml', project_dir)[0]
        datapath = mypath + '/data/HDF5/ChinaStockData.h5'
        print(len(code_df))
        for index, row in code_df.iterrows():
            ts_code = row['ts_code']
            mongo = MongoBase(db='ChinaStockData', collection=ts_code)
            results = mongo.collection.find()
            df = pd.DataFrame(list(results))
            del df['_id']
            df.to_hdf(datapath, ts_code)
            mongo.closeDB()
            # time.sleep(0.3)
            print(index)




if __name__ == '__main__':
    T = TushareHDF5()
    T.ReadfromMongoDB()




