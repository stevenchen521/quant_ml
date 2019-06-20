import talib
from rqalpha.api import *
from rqalpha import run_func
import pandas as pd
import numpy as np


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):

    # 选择我们感兴趣的股票
    # context.s1 = "000002.XSHE"
    # context.s1 = "000060.XSHE"
    context.TICKER = "002415.XSHE"
    # context.s2 = "601988.XSHG"
    # context.s3 = "000068.XSHE"
    context.stocks = [context.TICKER]
    # context.stocks = [context.s1, context.s2, context.s3]
    # context.stocks = pd.read_csv("ticker_list.csv").ticker.tolist()
    context.TIME_PERIOD = 14
    context.HIGH_RSI = 65
    context.LOW_RSI = 35
    context.ORDER_PERCENT = 0.9
    context.VOL_PERIOD = 60



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
        prices = history_bars(stock, context.TIME_PERIOD+2, '1d', 'close')
        # avg_price = history_bars(stock, 1, '1d', ['open', 'high', 'low', 'close'])
        # avg_price = np.mean(list(avg_price[0]))
        volumes = history_bars(stock, context.VOL_PERIOD, '1d', 'volume')
        avg_volume = np.mean(volumes)
        p_vol = volumes[-1] / avg_volume
        price_change = (prices[-1] - prices[-2]) / prices[-2]
        if (p_vol > 2 and abs(price_change) > 0.03) or (p_vol > 1.5 and abs(price_change) > 0.05) \
                or (p_vol > 3 and abs(price_change) > 0.015) or (abs(price_change) >= 0.1):
            Thrust = 1
        else:
            Thrust = 0

        if Thrust == 1 and price_change > 0:
            Thrust_up = 1
            Thrust_down = 0
        elif Thrust == 1 and price_change < 0:
            Thrust_up = 0
            Thrust_down = 1
        else:
            Thrust_up = 0
            Thrust_down = 0


        # 用Talib计算RSI值
        # rsi_data_today = talib.RSI(prices, timeperiod=context.TIME_PERIOD)[-1]
        # rsi_data_last_day = talib.RSI(prices, timeperiod=context.TIME_PERIOD)[-2]
        plot("Thrust_up", Thrust_up)
        plot("Thrust_down", Thrust_down)
        plot("close", prices[-1])

        cur_position = context.portfolio.positions[stock].quantity
        # 用剩余现金的30%来购买新的股票
        target_available_cash = context.portfolio.cash * context.ORDER_PERCENT

        # 当Thrust == 1以及 价格变化 > 0  Thrust up: buy target cash
        if Thrust_up == 1:
            order_value(stock, target_available_cash)

        # 当Thrust == 1以及 价格变化 < 0  Thrust down: clear position
        if Thrust_down == 1 and cur_position > 0 :
            logger.info("target available cash to order: " + str(target_available_cash))
            # 如果剩余的现金不够一手 - 100shares，那么会被ricequant 的order management system reject掉
            order_target_percent(stock, 0)


config = {
  "base": {
    "data_bundle_path": "D:\\rqalpha\data_bundle\\bundle",
    "start_date": "2011-01-01",
    "end_date": "2019-06-18",
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
      "output_file": "combine_RSI_ma_slope.pkl"
    }
  }
}

# 您可以指定您要传递的参数
# run_func(init=init, before_trading=before_trading, handle_bar=handle_bar, config=config)

# 如果你的函数命名是按照 API 规范来，则可以直接按照以下方式来运行
run_func(**globals())
from rqalpha_backtest.Base import pickle_to_excel
pickle_to_excel(pickle_path='combine_RSI_ma_slope.pkl')

