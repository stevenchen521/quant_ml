# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pandas as pd
import backtrader as bt
from backtrader.feeds import GenericCSVData
import datetime
import numpy as np

from back_testing.customize_function.mock_function import Cerebro, BacktestSummary
from back_testing.customize_function.customized_analyzer import Transactions
from back_testing.customize_function.Multi_datafeed_test import observers
import os

class GenericCSV_OTri(GenericCSVData):
    lines = ('OTri', 'Tri')
    params = (
        ('fromdate', datetime.datetime(2016, 8, 17)),
        ('todate', datetime.datetime(2019, 1, 23)),
        ('OTri', 7),
        ('Tri', 8)
    )


class PandasData(bt.feeds.PandasData):
    lines = ('Tri',)
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
        ('datetime', 'date'),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        ('Tri', 'Tri')
    )

class MyStrategy(bt.Strategy):
    ## trade_para first is tp_xu, second is tp_windowing
    params = (
        ('fromdate', datetime.datetime(2008, 1, 1)),
        ('todate', datetime.datetime(2019, 1, 23))
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # self.df = pd.DataFrame(columns=['Date', 'Buy_thres', 'Sell_thres', 'Action', 'Price',
        #                                 'Shares', 'Value', 'Commission', 'Gross', 'Net_profit'])
        # To keep track of pending orders and buy price/commission
        self.count = 0
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.order_dict = {}
        self.startcash = self.broker.getvalue()
        self.close_type = "None"
        # btind.SMA(self.data.tp_score, period=1, subplot=True)
        # btind.SMA(self.data.vp, period=1, subplot=True)
        # btind.SMA(self.data.q_g, period=1, subplot=True)


    def notify(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')



    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %(trade.pnl, trade.pnlcomm))

        # self.df['Gross'].loc[self.count-1] = trade.pnl
        # self.df['Net_profit'].loc[self.count-1] = trade.pnlcomm

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])
        # if self.order:
        #     return
        if self.datetime.datetime(ago=0) > datetime.datetime(2008, 1, 1):
            if not self.position: # not in the market
                # Not yet ... we MIGHT BUY if ...
                if (self.data.Tri[0] >= 0.8) and (self.data.Tri[-1] < 0.8):
                    # amount_to_invest = (self.p.order_pct * self.broker.cash)
                    # self.size = int(amount_to_invest / self.data.close)
                    self.order = self.buy(size=100)
                    # self.close_type = "None"

## the most import part of rolling stop loss
## compare current close price with the stored one, if current one is greater , replace the stored one

            if self.position:  # in the market
                # Not yet ... we will sell if ...
                if (self.data.Tri[0] < 0.5) and (self.data.Tri[-1] >= 0.5):
                    # amount_to_invest = (self.p.order_pct * self.broker.cash)
                    # self.size = int(amount_to_invest / self.data.close)
                    self.order = self.sell(size=100)
                else:
                    pass
            else:
                pass
        else:
            pass


    class BuySellObserver(observers.Observer):
        # alias = ('CashValue',)
        lines = ('buy', 'sell')
        # plotinfo = dict(plot=True, subplot=True)
        def next(self):
            self.lines.buy[0] = self._owner.broker.getcash()
            self.lines.sell[0] = self._owner.broker.getvalue()


    def stop(self):
        pnl = round(self.broker.getvalue() - self.startcash, 2)
        print('Final PnL: {}'.format(pnl))



def runstarts():
    import os
    import re
    project_dir = os.getcwd()
    mypath = re.findall(r'.*quant_ml', project_dir)[0]
    # file_name = "600276SH_for_backtest"
    file_name = 'DualAttnRNN_SH_index_all_for_backtest'

    data_path = mypath + "/back_testing/data/{}.csv".format(file_name)
    summary_path = mypath + "/back_testing/data_mining/summary_excel/{}_summary.xlsx".format(file_name)
    ticker_data_path = data_path
    commission = 0.002
    cerebro = Cerebro(maxcpus=2)
    # Add a strategy
    cerebro.addstrategy(MyStrategy)
    # Fetch_raw_data(ticker_data_path)
    df = pd.read_csv(ticker_data_path)
    data = GenericCSV_OTri(dataname=ticker_data_path,)
    # Add the Data Feed to Cerebro
    cerebro.adddata(data, name=file_name)
    cerebro.addanalyzer(bt.analyzers.PyFolio)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addanalyzer(Transactions)
    cerebro.broker.setcash(10000000)
    # cerebro.addsizer(bt.sizers.FixedSize, stake=10000)
    # Set the commission
    cerebro.broker.setcommission(commission=commission)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    # strat = results[0]
    cerebro.plot()
    '''
    Three input for BecktestSummary:
    results: result of cerebro.run()
    commission rate
    df: the input dataframe, backtesting dataframe
    save_path: the path to save the summary path
    groupby_list: when using optstrategy, add this para, which is the para name list we want to optimize
    '''
    Backtest_summary = BacktestSummary(results=results,
                                       input_df=df,
                                       commission=0.002,
                                       save_path=summary_path,
                                       groupby_list=None)
    Backtest_summary.analyze_and_save_single()



if __name__ == '__main__':
    runstarts()
