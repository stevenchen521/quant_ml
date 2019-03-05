
input_selector = '$'

strategy_nasdaq = {
    'module': 'base.env.pre_process',
    'file_path': '../../data/nasdaq.csv',
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': ['rsi_$close_14', 'macd_$close_12_26_9', 'stoch_$high_$low_$close_14_3', 'trend_$close_5_5_20'],
    'post_analyze': 'PostAnalyzeDefault',
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
    'analyze': ['rsi_$close_14', 'macd_$close_12_26_9', 'trend_$close_5_5_20'],
    'post_analyze': 'PostAnalyzeDefault',
}


active_stragery = strategy_nasdaq