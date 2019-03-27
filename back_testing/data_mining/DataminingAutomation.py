import math
import base.env.pre_process as pre_process
from back_testing.data_mining.dm_pre_process import strategy_data_mining
from sklearn.preprocessing import MinMaxScaler
from back_testing.common.common import BackTesting
import pandas as pd


def bt_strategy_setup():
    from back_testing.data_mining.strategy import MyStrategy
    return MyStrategy

def bt_datafeed_setup():
    from back_testing.data_mining.strategy import PandasData
    return PandasData


class DataMiningAutomation(object):
    def __init__(self, delta, step, strategy, start_date, end_date, bt_strategy_setup, bt_datafeed_setup):
        self._strategy_dict = self.manipulate_strategy(strategy, delta, step)
        self._start_date = start_date
        self._end_date = end_date
        self._bt_strategy = bt_strategy_setup()
        self._bt_datafeed = bt_datafeed_setup()

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

            # original input rank
            parameters_rank = rank(parameters)
            for parameter in parameters:
                # for every parameter, calculate the low and high boundaries.
                block = int((parameter / 2) * delta)
                boundary_high = parameter + block
                boundary_low = 1 if parameter - block <= 0 else parameter - block
                # get the parameter variable range.
                parameters_range.append((boundary_low, boundary_high, step))

            # calculate the list size of parameters compositions
            para_list_size = 1
            for range_para in parameters_range:
                para_list_size *= len(range(range_para[0], range_para[1], range_para[2]))

            # initialize the list for every parameters composition
            parameters_list = list()
            for _ in range(para_list_size):
                parameters_list.append(list())

            # iterate all the parameters compositions and put them to the list
            for range_para in parameters_range:
                cursor = 0
                while cursor < len(parameters_list):
                    para_val = [val for val in range(range_para[0], range_para[1], range_para[2])]
                    # for idx, val in enumerate(range(range_para[0], range_para[1], range_para[2])):
                    for val in para_val:
                        parameters_list[cursor].append(val)
                        cursor += 1

            # for every candidate in the list, calculate the rank, which should be the same with the original input.
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


    @staticmethod
    def preprocess_for_backtest(df, add_col=None):
        df['date'] = df.index
        df['date'] = df['date'].apply(
            lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
        df['openinterest'] = 0
        basic_col = ['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
        if add_col:
            col_order = basic_col + add_col
        else:
            col_order = basic_col
        df = df[col_order]
        df.columns = [x.lower() for x in list(df.columns) if x in basic_col] + add_col
        df.dropna(how="any", inplace=True)
        df.index = range(len(df))
        # df.set_index(['date'], inplace=True
        return df


    def format_data_dict(self):
        data_dict = {}
        for key, value in self._strategy_dict.items():
            dates, _, _, post_frames = \
                pre_process.ProcessStrategy(  # action_fetch, action_pre_analyze, action_analyze, action_post_analyze,
                    ['SH_index'], self._start_date, self._end_date, MinMaxScaler(), value).process()
            df_temp = post_frames['SH_index']
            df_temp = self.preprocess_for_backtest(df_temp, add_col=['Tri'])
            data_dict[key] = df_temp
        return data_dict


    def process(self):
        df_dict = self.format_data_dict()
        back_testing = BackTesting(MyStrategy=self._bt_strategy,
                                   Datafeed=self._bt_datafeed,
                                   input_dict=df_dict,
                                   domain_name="SH_index",
                                   save_input_dict=True,
                                   summary_path='summary.xlsx')
        back_testing.backtest()


if __name__ == '__main__':
    D = DataMiningAutomation(delta=2,
                             step=1,
                             strategy=strategy_data_mining,
                             start_date="2008-01-01",
                             end_date="2019-02-01",
                             bt_strategy_setup=bt_strategy_setup,
                             bt_datafeed_setup=bt_datafeed_setup)
    D.process()


