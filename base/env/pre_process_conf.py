file_path = "../../data/nasdaq.csv"
# file_path = "../../data/SH_index.csv"
# file_path = "../../data/600276SH.csv"
# analysis = []
# analysis = ['Close_rsi_14', 'Close_macd_12_26_9']


strategy_nasdaq = {
    'module': 'base.env.pre_process',
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': ['close_rsi_14', 'close_macd_12_26_9', 'close_trend_5_5_20'],
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
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': ['close_rsi_14', 'close_macd_12_26_9', 'close_trend_5_5_20'],
    'post_analyze': 'PostAnalyzeDefault',
}


active_stragery = strategy_nasdaq