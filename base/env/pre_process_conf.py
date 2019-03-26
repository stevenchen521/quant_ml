
import os

def get_strategy_analyze(strategy):
    return strategy['analyze']


def get_folder(file):
    return os.sep.join(file.split(os.sep)[0:-1])


"""
analyze format: 'method'|'input columns'|'method parameters'
    i.e. macd|close|12_26_9: indicator MACD on the close price with parameter 12, 26, 9
    i.e. stoch|high_low_close|14_3: indicator STOCH on the 'high, low, close' with parameter 14, 3
"""

strategy_play = {
    'name':'strategy_play',
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



strategy_nasdaq = {
    'name':'strategy_nasdaq',
    'module': 'base.env.pre_process',
    'source': "{}/../../data/nasdaq.csv".format(get_folder(__file__)),
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
                'willr|high_low_close|14',
                'trend|close|5_5_10'],
    'post_analyze': 'PostAnalyzeNASDAQ',
    'label': 'trend|close|5_5_10'
    # 'post_analyze': 'PostAnalyzeDefault',
}

'''
close trend para:
first one: moving average length,  if we set small, indicator will become more sensitive
second one: trend bar length,
third one: future bar length, focus on return in future length
'''
strategy_SH_index = {
    'name': 'strategy_SH_index',
    'module': 'base.env.pre_process',
    'source': "{}/../../data/SH_index.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
                # 'ma|close|5_0',
                # 'ma|close|5_1',
                # 'ma|close|5_2',
                # 'rsi|close|14',
                # 'macd|close|12_26_9',
                # 'minus_dm|high_low|14', 'plus_dm|high_low|14', 'adx|high_low_close|14', # Directional Movement Index(DMI)
                'stoch|high_low_close|14_3_0_3_0',
                # 'stoch|high_low_close|14_5_0_5_0',
                # 'willr|high_low_close|14',
                # 'apo|close|12_26_0',
                'trend_backward|close|5_5_3',
                'trend_backward|close|5_5_2',
                # 'trend_backward|close|10_3_10',
                # 'trend_backward|close|10_3_5',
                # 'roc|close|5',
                # 'rocp|close|5',
                'bop|open_high_low_close',
                'obv|close_volume',
                'trend|close|5_5_3'],
    'post_analyze': 'PostAnalyzeNASDAQ',
    'label': 'trend|close|5_5_3'
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
# active_stragery = 'base.env.pre_process_conf.strategy_SH_index'

