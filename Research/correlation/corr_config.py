from base.env.pre_process_conf import get_folder

research_config = {
    'name': 'data_mining',
    'module': 'Research.correlation.correlation',
    'source': "{}/../../data/SH_index_all.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVResearch',
    'pre_analyze': 'PreAnalyzeResearch',
    'analyze': [
        # 'roc|close|20',
        # 'trend_backward|close|10_5_2',
        'trend_backward|open|10_5_10',
        'trend_backward|on|10_5_10',
        'trend_backward|6m|10_5_10',
        # 'trend_backward|IS_EPS|10_5_2',
        'trend_backward|nasclose|10_5_10',

        'trend|close|10_5_20',
        'bop|open_high_low_close',
        # 'obv|close_volume',
        # 'ad|high_low_close_volume',
        # 'adosc|high_low_close_volume|3_10',
        'rsi|close|14',
        'stoch|high_low_close|14_3',
                ],
    'post_analyze': 'PostAnalyzeResearch',
    'label': 'trend|close|10_5_20'
}

active_config = research_config
