from base.env.pre_process_conf import get_folder

research_config = {
    'name': 'data_mining',
    'module': 'back_testing.data_mining.dm_pre_process',
    'source': "{}/../../data/SH_index.csv".format(get_folder(__file__)),
    'fetch': 'FetchCSVSingleDM',
    'pre_analyze': 'PreAnalyzeDataMining',
    'analyze': [
        'trend_backward|close|10_5_2',
        'ma|close|50',
        'trend|close|10_5_20',
        'bop|open_high_low_close',
        'obv|close_volume',
        'rsi|close|14'
                ],
    'post_analyze': 'PostAnalyzeDataMining',
    'label': 'trend|close|10_5_20'
}

active_config = research_config
