import talib
from rqalpha.api import *
from rqalpha import run_func
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):

    # 选择我们感兴趣的股票
    context.s1 = "002475.XSHE"
    # context.s2 = "601988.XSHG"
    # context.s3 = "000068.XSHE"
    context.stocks = [context.s1]
    # context.stocks = pd.read_csv("ticker_list.csv").ticker.tolist()
    context.TIME_PERIOD = 14
    context.HIGH_RSI = 70
    context.LOW_RSI = 30
    context.ORDER_PERCENT = 0.5


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    # 开始编写你的主要的算法逻辑

    # bar_dict[order_book_id] 可以拿到某个证券的bar信息
    # context.portfolio 可以拿到现在的投资组合状态信息

    # 使用order_shares(id_or_ins, amount)方法进行落单

    # 对我们选中的股票集合进行loop，运算每一只股票的RSI数值
    for stock in context.stocks:
        # 读取历史数据
        print(stock)
        prices = history_bars(stock, 50, '1d', 'close')

        # 用Talib计算RSI值
        rsi_data = talib.RSI(prices, timeperiod=context.TIME_PERIOD)
        rsi_data = rsi_data[~np.isnan(rsi_data)]
        max_position = argrelextrema(rsi_data, np.greater)[-1][-1]
        min_position =argrelextrema(rsi_data, np.less)[-1][-1]

        if max_position > min_position:
            max_rsi = rsi_data[max_position]
            min_rsi = 0

        else:
            min_rsi = rsi_data[min_position]
            max_rsi = 0

        rsi_data_today = rsi_data[-1]
        # rsi_data_last_day = rsi_data[-2]
        plot('RSI', rsi_data_today)

        cur_position = context.portfolio.positions[stock].quantity
        # 用剩余现金的50%来购买新的股票
        target_available_cash = context.portfolio.cash * context.ORDER_PERCENT

        # 当RSI大于设置的上限阀值，清仓该股票
        if min_rsi > 0 and rsi_data_today - min_rsi > 15:
            order_target_value(stock, 0.5)
            plot('sell', 0)

        # 当RSI小于设置的下限阀值，用剩余cash的一定比例补仓该股
        if max_rsi > 0 and max_rsi - rsi_data_today > 15:
            logger.info("target available cash to order: " + str(target_available_cash))
            # 如果剩余的现金不够一手 - 100shares，那么会被ricequant 的order management system reject掉
            order_value(stock, target_available_cash)
            plot('buy', 100)


config = {
  "base": {
    "data_bundle_path": "D:\\rqalpha\data_bundle\\bundle",
    "start_date": "2017-06-01",
    "end_date": "2019-06-10",
    "benchmark": "000300.XSHG",
    "accounts": {
        "stock": 10000000
    }
  },
  "extra": {
    "log_level": "verbose",
  },
  "mod": {
    "sys_analyser": {
      "enabled": True,
      "plot": True,
      "output_file": "Portfolio_RSI.pkl"
    }
  }
}

# 您可以指定您要传递的参数
# run_func(init=init, before_trading=before_trading, handle_bar=handle_bar, config=config)

# 如果你的函数命名是按照 API 规范来，则可以直接按照以下方式来运行
run_func(**globals())
from rqalpha_backtest.Base import pickle_to_excel
pickle_to_excel(pickle_path='Portfolio_RSI.pkl')