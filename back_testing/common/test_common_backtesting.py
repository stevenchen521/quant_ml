from unittest import TestCase
from back_testing.common.common import BackTesting
from back_testing.OTr_back_test import MyStrategy
import pandas as pd

class TestBacktesting(TestCase):
    def test_save(self):
        self.fail()

    def test_backtest(self):
        df1 = pd.read_csv('../../back_testing/data/nasdaq_for_backtest.csv')
        dict_test = {'nasdaq_for_backtest_shibor1': df1,
                     'nasdaq_for_backtest_KDJ14_5': df1}
        Backtesting = BackTesting(
            MyStrategy=MyStrategy,
            input_dict=dict_test,
            domain_name='strategy_nasdaq',
            save_input_dict=True,
            target_col='OTri',
            label='Tri'
        )
        Backtesting.backtest()

