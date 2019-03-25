from unittest import TestCase
from back_testing.common.common import BackTesting, preprocess_for_backtest
from back_testing.OTr_back_test import MyStrategy
from back_testing.data_mining.strategy import PandasData
import pandas as pd


class TestBacktesting(TestCase):
    def test_save(self):
        self.fail()



    def test_pre_process(self):
        df = pd.read_csv('nasdaq.csv')
        df.columns = df.columns.str.lower()
        print (df.columns)


    def test_backtest(self):
        df1 = pd.read_csv('../../back_testing/data/nasdaq_for_backtest.csv')
        dict_test = {'nasdaq_for_backtest_shibor1': df1,
                     'nasdaq_for_backtest_KDJ14_5': df1}
        Backtesting = BackTesting(
            MyStrategy=MyStrategy,
            input_dict=dict_test,
            domain_name='strategy_nasdaq',
            save_input_dict=True,
            Datafeed=PandasData
        )
        Backtesting.backtest()

