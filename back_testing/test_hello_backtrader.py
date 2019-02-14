from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import backtrader as bt
import os
import datetime
import pandas as pd
# import sys
from back_testing.hello_backtrader import QuickstartStrategy
from back_testing.hello_backtrader import DummyInd


class TestBacktrader(unittest.TestCase):

    def test_backtrader(self):
        # Create a cerebro entity
        cerebro = bt.Cerebro()

        # Add a strategy
        cerebro.addstrategy(QuickstartStrategy)

        # Datas are in a subfolder of the samples. Need to find where the script is
        # because it could have been called from anywhere
        # modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
        datapath = os.path.join('./orcl.csv')

        # Create a Data Feed
        data = bt.feeds.YahooFinanceCSVData(
            dataname=datapath,
            # Do not pass values before this date
            fromdate=datetime.datetime(2000, 1, 1),
            # Do not pass values before this date
            todate=datetime.datetime(2000, 12, 31),
            # Do not pass values after this date
            reverse=False)

        # data = bt.feeds.GenericCSVData(
        #     dataname=datapath,
        #     # Do not pass values before this date
        #     fromdate=datetime.datetime(2000, 1, 1),
        #     # Do not pass values before this date
        #     todate=datetime.datetime(2000, 12, 31),
        #     dtformat=('%Y-%m-%d'),
        #
        #     close=5,
        #
        #     )

        # Add the Data Feed to Cerebro
        cerebro.adddata(data)

        cerebro.addindicator(DummyInd)

        # Set our desired cash start
        cerebro.broker.setcash(100000.0)

        # Add a FixedSize sizer according to the stake
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)  # profit * 10 times

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Run over everything
        cerebro.run()

        # Print out the final result
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Plot the result
        cerebro.plot()

    def test_datafeed_pd(self):
        # Create a cerebro entity
        cerebro = bt.Cerebro()

        # Add a strategy
        cerebro.addstrategy(QuickstartStrategy)

        df_data = pd.read_csv('./orcl.csv',parse_dates=True,
                                index_col=0)

        data = bt.feeds.PandasData(dataname=df_data,close=4)

        # Add the Data Feed to Cerebro
        cerebro.adddata(data)

        # Set our desired cash start
        cerebro.broker.setcash(100000.0)

        # Add a FixedSize sizer according to the stake
        cerebro.addsizer(bt.sizers.FixedSize, stake=10)  # profit * 10 times

        # Set the commission - 0.1% ... divide by 100 to remove the %
        cerebro.broker.setcommission(commission=0.001)

        # Print out the starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Run over everything
        cerebro.run()

        # Print out the final result
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Plot the result
        cerebro.plot()

if __name__ == '__main__':
    unittest.main()