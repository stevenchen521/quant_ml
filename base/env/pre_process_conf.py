
# input_selector = '$'

"""
analyze format: 'method'|'input columns'|'method parameters'
    i.e. macd|close|12_26_9: indicator MACD on the close price with parameter 12, 26, 9
    i.e. stoch|high_low_close|14_3: indicator STOCH on the 'high, low, close' with parameter 14, 3
"""


strategy_nasdaq = {
    'module': 'base.env.pre_process',
    'source': '../../data/nasdaq.csv',
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
                'ma|close|5_0',
                'ma|close|5_1',
                'ma|close|5_2',
                'rsi|close|14',
                'macd|close|12_26_9',
                'minus_dm|high_low|14', 'plus_dm|high_low|14', 'adx|high_low_close|14', # Directional Movement Index(DMI)
                'stoch|high_low_close|14_3',
                'trend|close|5_5_20'],
    'post_analyze': 'PostAnalyzeNASDAQ',
    'label': 'trend_5_5_20'
    # 'post_analyze': 'PostAnalyzeDefault',
}

'''
close trend para:
first one: moving average length,  if we set small, indicator will become more sensitive
second one: trend bar length,
third one: future bar length, focus on return in future length
'''
strategy_SH_index = {
    'module': 'base.env.pre_process',
    'source': '../../data/SH_index.csv',
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
                # 'ma|close|10_0', # sma
                # 'ma|close|10_1', # ema
                # 'ma|close|10_2', # wma
                # 'ma|close|10_3', # dema
                # 'ma|close|10_4', # tema
                # 'rsi|close|14',
                # 'macd|close|12_26_9',
                # 'minus_dm|high_low|14', 'plus_dm|high_low|14', 'adx|high_low_close|14', # Directional Movement Index(DMI)
                'stoch|high_low_close|14_5',    # key indicator
                'trend|close|5_5_20'],
    'post_analyze': 'PostAnalyzeNASDAQ',
    'label': 'trend_5_5_20'
    # 'post_analyze': 'PostAnalyzeDefault',
}

#
# strategy_SH_index = {
#     'module': 'base.env.pre_process',
#     'source': '../../data/SH_index.csv',
#     'fetch': 'FetchCSVSingle',
#     'pre_analyze': 'PreAnalyzeDefault',
#     'analyze': ['rsi|close|14', 'macd|close|12_26_9', 'trend|close|5_5_20'],
#     'post_analyze': 'PostAnalyzeDefault',
# }


# active_stragery = strategy_nasdaq
active_stragery = strategy_SH_index