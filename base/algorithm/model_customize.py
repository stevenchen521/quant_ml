import tensorflow as tf
import numpy as np
import json

from abc import abstractmethod
from helper import data_ploter
from tensorflow.contrib import rnn
from helper.data_logger import generate_algorithm_logger
import math
from functools import reduce
from base.algorithm.model import BaseSLTFModel

class BaseSLTFModel_folk_pre_1Dre_raw_return(BaseSLTFModel):
    def __init__(self,session, env, **options):
        super(BaseSLTFModel_folk_pre_1Dre_raw_return, self).__init__(session, env, **options)

    def eval_and_plot(self):
        def _to_price(list):
            ### return to price
            func = np.vectorize(lambda x: x/100 + 1)
            ### return to daily change
            list = func(list)
            label1 = [reduce(lambda x, y: x * y, list[:i + 1]).tolist() for i in range(len(list))]
            price_list = np.array([[i[0] * first_close] for i in label1])
            return price_list

        x, label = self.env.get_test_data()
        # price = self.env.original_frame
        # add new_code, scale the label and predict back
        first_index = self.env.e_data_indices[0]-1
        first_close = self.env.origin_frames['600036']['close'].iloc[first_index]

        # mean = self.env.scaler[0].mean_[-1]
        # std = math.sqrt(self.env.scaler[0].var_[-1])
        label_price = _to_price(label)

        y = self.predict(x)
        y_price = _to_price(y)
        with open(self.save_path + '_label_price.json', mode='w') as fp:
            json.dump(label_price.tolist(), fp, indent=True)

        with open(self.save_path + '_label_daily_change.json', mode='w') as fp:
            json.dump(label.tolist(), fp, indent=True)

        with open(self.save_path + '_y_price.json', mode='w') as fp:
            json.dump(y_price.tolist(), fp, indent=True)

        with open(self.save_path + '_y_daily_change.json', mode='w') as fp:
            json.dump(y.tolist(), fp, indent=True)
        data_ploter.plot_stock_series(self.env.codes,
                                      y_price,
                                      label_price,
                                      self.save_path + '_price')
        data_ploter.plot_stock_series(self.env.codes,
                                      y,
                                      label,
                                      self.save_path + '_daily_change')


