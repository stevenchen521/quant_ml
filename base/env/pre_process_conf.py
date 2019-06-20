
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
strategy_DualAttn = {
    'name': 'strategy_SH_index',
    'module': 'base.env.pre_process',
    'source': "{}/../../data/SH_index_all.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
        'stoch|high_low_close|14_3_0_3_0',  ##pos
        'trend_backward|close|10_5_5',
        'rsi|close|14',
        'roc|close|20',
        'FFT|close|3_6_9',
        'arima|close|5_1_0',
        'trend_backward|nasclose|5_5_10',  ## index
        'trend|close|10_5_10'],
    'label': 'trend|close|10_5_10',
    'post_analyze': 'PostAnalyzeDefault'
}


strategy_SH_index2 = {
    'name': 'strategy_SH_index',
    'module': 'base.env.pre_process',
    'source': "{}/../../data/SH_index_all.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
        'stoch|high_low_close|14_3_0_3_0',  ##pos
        'trend_backward|close|10_5_3',
        'rsi|close|14',
        'roc|close|20',
        'trend_backward|on|10_5_2',  # neg or no corr
        'trend_backward|6m|10_5_2',
        # 'bop|open_high_low_close',
        # 'obv|close_volume',
        'trend_backward|nasclose|10_5_2',  ## index
        'trend|close|10_5_20'],
    'post_analyze': 'PostAnalyzeNASDAQ',
    'label': 'trend|close|10_5_20'
    # 'post_analyze': 'PostAnalyzeDefault',
}


#
research_config = {
    'name': 'data_mining',
    'module': 'base.env.pre_process',
    'source': "{}/../../data/SH_index_all.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
        'roc|close|20',  ## pos
        'stoch|high_low_close|14_3_0_3_0', ##pos
        'trend_backward|close|10_5_2',  ## pos
        'trend_backward|on|10_5_2',  # no corr
        'trend_backward|6m|10_5_2',   ## no corr
        'rsi|close|14',
        # 'bop|open_high_low_close',
        # 'obv|close_volume',
        # 'ad|high_low_close_volume',
        'stoch|high_low_close|14_3',
        'trend_backward|nasclose|10_5_2', ## index
        'trend|close|5_5_20', ## target
                ],
    'post_analyze': 'PostAnalyzeNASDAQ',
    'label': 'trend|close|5_5_20'
}



MILSTM_strategy = {
    'name': 'strategy_SH_index',
    'module': 'base.env.pre_process',
    'source': "{}/../../data/SH_index_all.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': [
        'stoch|high_low_close|14_3_0_3_0',  ##pos
        'trend_backward|close|10_5_10',
        'rsi|close|14',
        'roc|close|20',
        # 'trend_backward|on|10_5_2',  # neg or no corr
        # 'trend_backward|6m|10_5_2',
        # 'bop|open_high_low_close',
        # 'obv|close_volume',
        'trend_backward|nasclose|10_5_10',  ## index
        'trend|close|10_5_10'],
    'post_analyze': 'PostAnalyzeMIlstm',
    'label': 'trend|close|10_5_10'
    # 'post_analyze': 'PostAnalyzeNASDAQ',
}
# active_stragery = strategy_nasdaq
active_stragery = strategy_DualAttn
# active_stragery = 'base.env.pre_process_conf.strategy_SH_index'

