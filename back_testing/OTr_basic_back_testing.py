from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
from backtrader.feeds import GenericCSVData, DataBase
# Import the backtrader platform
import backtrader as bt

class GenericCSV_vp(GenericCSVData):
    lines = ('OTri',)
    params = (
        ('fromdate', datetime.datetime(2016, 11, 17)),
        ('todate', datetime.datetime(2019, 1, 28)),
        ('OTri', 7),
    )

class PandasData(DataBase):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''

    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', None),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('Open', -1),
        ('High', -1),
        ('Low', -1),
        ('Close', -1),
        ('Volume', -1),
        ('openinterest', -1),
        ('OTri', 7),
        ('Tri', 8),
    )

# Create a Stratey
class TestStrategy(bt.Strategy):

    params = (
        ('fromdate', datetime.datetime(2016, 11, 17)),
        ('todate', datetime.datetime(2019, 1, 28))
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders
        self.order = None

    def notify(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        # Write down: no pending order
        self.order = None


    def next(self):
        self.log('Close, %.2f, OTri, %.2f' % (self.dataclose[0], self.data.OTri[0]))
        # self.log('Close, %.2f' % (self.dataclose[0]))
        if self.order:
            return
        if not self.position:  # not in the market
            # Not yet ... we MIGHT BUY if ...
            if (self.data.OTri[0] > 0.5) and (self.data.OTri[-1] < 0.5):
            # if (self.data.tp_score[0] > 50) and (self.data.tp_score[-1] < 50):
                # amount_to_invest = (self.p.order_pct * self.broker.cash)
                # self.size = int(amount_to_invest / self.data.close)
                self.order = self.buy(size=100)
                # self.close_type = "None"

        ## the most import part of rolling stop loss
        ## compare current close price with the stored one, if current one ish greater , replace the stored one

        if self.position:  # in the market
            # Not yet ... we will sell if ...
            if (self.data.OTri[0] < 0.5) and (self.data.OTri[-1] > 0.5):
            # if (self.data.tp_score[0] < 50) and (self.data.tp_score[-1] > 50):
                # amount_to_invest = (self.p.order_pct * self.broker.cash)
                # self.size = int(amount_to_invest / self.data.close)
                self.order = self.sell(size=100)




def Fetch_raw_data(input_data):
    df1 = pd.read_csv(input_data)
    df1['Date'] = df1['Date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
    df1.dropna(how="any", inplace=True)
    df1.set_index(['Date'], inplace=True)
    df1['openinterest'] = 0
    col_order = ['Open', 'High', 'Low', 'Close', 'Volume', 'openinterest', 'OTri', 'Tri']
    df1 = df1[col_order]
    df1.to_csv(input_data, date_format='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)
    import sys
    mypath = os.path.dirname(sys.modules['__main__'].__file__)
    file_name = "nasdaq_for_backtest_processed"
    # file_name = "5 HK EQUITY"
    data_path = mypath + "/data/{}.csv".format(file_name)

    ticker_data_path = data_path
    commission = 0.002
    # Add a strategy
    data = GenericCSV_vp(dataname=ticker_data_path,)
    # Add the Data Feed to Cerebro
    cerebro.adddata(data, name=file_name)

    # Create a Data Feed

    # dataframe['openinterest'] = 0
    Fetch_raw_data(ticker_data_path)
    # Add the Data Feed to Cerebro
    cerebro.adddata(data)
    # cerebro.addwriter(bt.WriterFile, csv=True)
    # Set our desired cash start
    cerebro.broker.setcash(1000000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # Plot the result
    cerebro.plot()

