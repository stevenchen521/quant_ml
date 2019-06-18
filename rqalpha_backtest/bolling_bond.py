import numpy as np
import talib
from rqalpha.api import *
from rqalpha import run_file
from rqalpha import run_func


def init(context):
    # context内引入全局变量s1
    context.s1 = "000001.XSHE"  # 测试下 000001
    context.AvgLen = 3  # 均线周期
    context.Disp = 16  # 布林平移参数
    context.SDLen = 12  # 布林标准差周期参数
    context.SDev = 3  # 布林通道倍数参数
    context.observetime = 50
    subscribe(context.s1)
    # 实时打印日志
    logger.info("RunInfo: {}".format(context.run_info))
    # 初始化context.last_main_symbol
    context.last_main_symbol = context.s1
    # 将判断主力是否更新的flag初始设置为False
    context.main_changed = False
    context.ORDER_PERCENT = 0.8
    context.TIME_PERIOD = 14


def before_trading(context):  # 每天开盘前判断主力合约是否更新。
    context.s1 = "000001.XSHE"
    # if context.last_main_symbol != context.s1:
    #     subscribe(context.s1)
    #     # 如果更新了，设置main_changed这个flag为True
    #     context.main_changed = True


# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):
    prices = history_bars(context.s1, context.observetime + 1, '1d', ['close'])
    Close = prices['close']
    AvgVal = talib.SMA(Close, context.AvgLen)  # 计算均线
    SDmult = np.std(Close[-context.SDLen:]) * context.SDev  # 计算标准差 布林标准差周期参数 * 倍数参数， 95% 置信区间
    DisTop = AvgVal[-context.Disp] + SDmult  # 平移后的上轨
    DisBottom = AvgVal[-context.Disp] - SDmult  # 平移后的下轨
    rsi_today = talib.RSI(Close, timeperiod=context.TIME_PERIOD)[-1]

    plot("Top bolling", DisTop)
    plot("Avg bolling", AvgVal[-context.Disp])
    plot("Bottom bolling", DisBottom)
    # plot("price", Close[-1])
    plot("rsi", rsi_today)
    # 期货
    # sell_qty = context.portfolio.positions[context.s1].sell_quantity
    # buy_qty = context.portfolio.positions[context.s1].buy_quantity

    #股票
    quantity = context.portfolio.positions[context.s1].quantity


    # 合约才需要变化
    # # 检测当前主力合约，如果发生变化，先对原有合约平仓
    # if context.main_changed:
    #     # 主力合约发变化，平仓
    #     print('Symbol Changed! before:', context.last_main_symbol, 'after: ', context.s1, context.now)
    #     # 空单平仓
    #     if context.portfolio.positions[context.last_main_symbol].sell_quantity != 0:
    #         buy_close(context.last_main_symbol, 1)
    #         print('close short:', context.now)
    #     # 多头平仓
    #     if context.portfolio.positions[context.last_main_symbol].buy_quantity != 0:
    #         sell_close(context.last_main_symbol, 1)
    #         print('close long', context.now)
    #     context.main_changed = False
    #     context.last_main_symbol = context.s1


    # 上穿下穿的影响
    # if prices['close'][-1] >= DisTop and prices['close'][-2] < DisTop: # close上穿上轨
    #     cur_position = context.portfolio.positions[context.s1].quantity
    #     if cur_position > 0:
    #         order_target_percent(context.s1, 0)
    #         print (context.portfolio)
    #     else:
    #         pass
    # if prices['close'][-1] <= DisBottom and prices['close'][-2] > DisBottom:  # close下穿下轨
    #     target_available_cash = context.portfolio.cash * context.ORDER_PERCENT
    #     order_target_value(context.s1, target_available_cash)
    #     print(context.portfolio)
    #
    # pure value comparation
    if prices['close'][-1] >= DisTop: # close上穿上轨
        cur_position = context.portfolio.positions[context.s1].quantity
        if cur_position > 0:
            order_target_percent(context.s1, 0)
            print (context.portfolio)
        else:
            pass

    if prices['close'][-1] <= DisBottom:  # close下穿下轨
        target_available_cash = context.portfolio.cash * context.ORDER_PERCENT
        order_target_value(context.s1, target_available_cash)
        print(context.portfolio)






    # # 空头
    # if buy_qty == 0 and sell_qty == 0:
    #     if prices['low'][-1] <= DisBottom:
    #         sell_open(context.s1, 1)
    # if sell_qty != 0:
    #     if prices['high'][-1] >= DisTop:
    #         buy_close(context.s1, 1)

config = {
  "base": {
    "data_bundle_path": "D:\\rqalpha\data_bundle\\bundle",
    "start_date": "2010-01-01",
    "end_date": "2019-06-10",
    "benchmark": "000300.XSHG",
    "accounts": {
        "stock": 1000000
    }
  },
  "extra": {
    "log_level": "verbose",
  },
  "mod": {
    "sys_analyser": {
      "enabled": True,
      "plot": True,
      "output_file": "bolling_bond.pkl"
    }
  }
}

# 您可以指定您要传递的参数
# run_func(init=init, before_trading=before_trading, handle_bar=handle_bar, config=config)

# 如果你的函数命名是按照 API 规范来，则可以直接按照以下方式来运行
run_func(**globals())


