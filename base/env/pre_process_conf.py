file_path = "../../data/nasdaq.csv"
# analysis = []
# analysis = ['Close_rsi_14', 'Close_macd_12_26_9']


strategy_nasdaq = {
    'module': 'base.env.pre_process',
    'fetch': 'FetchCSVSingle',
    'pre_analyze': 'PreAnalyzeDefault',
    'analyze': ['close_rsi_14', 'close_macd_12_26_9', 'close_stoch_5_3_3', 'close_trend_15_5_3'],
    'post_analyze': 'PostAnalyzeNASDAQ',
    # 'post_analyze': 'PostAnalyzeDefault',
}


active_stragery = strategy_nasdaq