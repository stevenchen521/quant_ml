import math
import base.env.pre_process as pre_process
from back_testing.data_mining.dm_pre_process import strategy_data_mining
from sklearn.preprocessing import MinMaxScaler
from back_testing.common.common import BackTesting
import pandas as pd


def bt_strategy_setup():
    from back_testing.OTr_back_test import MyStrategy
    return MyStrategy


class DataMiningAutomation(object):
    def __init__(self, delta, step, strategy, start_date, end_date, bt_strategy_setup):
        self._strategy_dict = self.manipulate_strategy(strategy, delta, step)
        self._start_date = start_date
        self._end_date = end_date
        self._bt_strategy_setup = bt_strategy_setup
        self._bt_strategy = bt_strategy_setup()

    def manipulate_strategy(self, strategy, delta, step):

        def rank(l):
            series_l = pd.Series(l).sort_values()
            result = []
            last_idx = None
            for idx in series_l.index:
                if idx > 0 and series_l[idx-1] == series_l[idx]:
                    result.append(last_idx)
                else:
                    result.append(idx)
                last_idx = result[len(result)-1]
            return result

        strategy_dict = dict()

        analyze_list = pre_process.get_strategy_element(strategy, "analyze")
        parameters_range = list()


        for analyze in analyze_list:
            # for every analyze
            analyze_parts = analyze.split("|")
            parameters_part = analyze_parts[len(analyze_parts) - 1]
            parameters = [int(para) for para in parameters_part.split("_")]
            parameters_rank = rank(parameters)
            for parameter in parameters:
                block = int((parameter / 2) * delta)
                boundary_high = parameter + block
                boundary_low = 1 if parameter - block <= 0 else parameter - block
                parameters_range.append((boundary_low, boundary_high, step))

            para_list_size = 1
            for range_para in parameters_range:
                para_list_size *= len(range(range_para[0], range_para[1], range_para[2]))

            # now we have the parameter range
            parameters_list = list()
            for _ in range(para_list_size):
                parameters_list.append(list())

            for range_para in parameters_range:
                cursor = 0
                while cursor < len(parameters_list):
                    para_val = [val for val in range(range_para[0], range_para[1], range_para[2])]
                    # for idx, val in enumerate(range(range_para[0], range_para[1], range_para[2])):
                    for val in para_val:
                        parameters_list[cursor].append(val)
                        cursor += 1

            parameters_list_final = list()
            for candidate in parameters_list:
                r = rank(candidate)
                if parameters_rank == r:
                    # parameters_list_final.append(candidate)
                    strategy_ouput = strategy.copy()
                    analyze_parts[len(analyze_parts)-1] = "_".join([str(val) for val in candidate])

                    analyze_str = "|".join(analyze_parts)

                    pre_process.set_strategy_element(strategy_ouput, "analyze", [analyze_str])
                    strategy_dict[analyze_str] = strategy_ouput

        return strategy_dict

    def format_data_dict(self):
        data_dict = {}
        for key, value in self._strategy_dict.items():
            dates, pre_frames, origin_frames, post_frames = \
                pre_process.ProcessStrategy(  # action_fetch, action_pre_analyze, action_analyze, action_post_analyze,
                    ['SH_index'], self._start_date, self._end_date, MinMaxScaler(), value).process()
            data_dict[key] = post_frames
        return data_dict


    def process(self):
        df_dict = self.format_data_dict()
        back_testing = BackTesting(MyStrategy=self._bt_strategy,
                                   input_dict=df_dict,
                                   domain_name="SH_index",
                                   save_input_dict=True,
                                   target_col='y',
                                   label='label')
        back_testing.backtest()

if __name__ == '__main__':
    D = DataMiningAutomation(delta=2,
                             step=1,
                             strategy=strategy_data_mining,
                             start_date="2008-01-01",
                             end_date="2019-02-01",
                             bt_strategy_setup= bt_strategy_setup)
    D.process()


