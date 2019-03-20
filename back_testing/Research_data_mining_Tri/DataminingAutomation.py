import base.env.pre_process as pre_process
from back_testing.Research_data_mining_Tri.pre_process_conf import strategy_dict
from sklearn.preprocessing import MinMaxScaler
from back_testing.common.common import BackTesting


def bt_strategy_setup():
    from back_testing.OTr_back_test import MyStrategy
    return MyStrategy


class DataMiningAutomation(object):
    def __init__(self, strategy_dict, start_date, end_date, bt_strategy_setup):
        self.strategy_dict = strategy_dict
        self.start_date = start_date
        self.end_date = end_date
        self._bt_strategy_setup = bt_strategy_setup
        self._bt_strategy = bt_strategy_setup()


    def format_data_dict(self):
        data_dict = {}
        for key, value in self.strategy_dict.items():
            dates, pre_frames, origin_frames, post_frames = \
                pre_process.ProcessStrategy(  # action_fetch, action_pre_analyze, action_analyze, action_post_analyze,
                    ['SH_index_all'], self.start_date, self.end_date, MinMaxScaler(), value).process()
            data_dict[key] = origin_frames
        return data_dict


    def process(self):
        df_dict = self.format_data_dict()
        back_testing = BackTesting(MyStrategy=self._bt_strategy,
                                   input_dict=df_dict,
                                   domain_name="SH_index_all",
                                   save_input_dict=True,
                                   target_col='y',
                                   label='label')
        back_testing.backtest()

if __name__ == '__main__':
    D = DataMiningAutomation(strategy_dict, "2008-01-01", "2019-02-01", bt_strategy_setup)
    D.process()


