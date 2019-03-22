import os
import tensorflow as tf
from back_testing.common.common import BackTesting

from algorithm import config
from base.env.market import Market
from checkpoints import CHECKPOINTS_DIR
from sklearn.preprocessing import MinMaxScaler
# from algorithm.SL.DualAttnRNN import Algorithm
from helper.args_parser import model_launcher_parser
from helper.util import get_timestamp, get_attribute
from base.env.pre_process_conf import active_stragery, get_strategy_analyze


"""
######## Market Setup #######
"""


def market_setup_sl(args, strategy):

    codes = ["SH_index"]
    market = args.market

    return Market(codes, start_date="2006-10-09", end_date="2019-02-27", **{
        "pre_process_strategy": strategy,
        "market": market,
        "use_sequence": True,
        "seq_length": 10,
        "scaler": MinMaxScaler(feature_range=(0, 1)),
        "mix_index_state": True,
        "training_data_ratio": 0.8,
    })


"""
######## Input Strategy Setup #######
"""


def input_strategy_setup(input_strategy):
    # strategy = get_attribute(active_stragery) if input_strategy is None else input_strategy
    result = list()
    strategy = active_stragery if input_strategy is None else input_strategy
    label = active_stragery['label']
    is_ori_label = True if label.find("|") == -1 else False
    analyze = get_strategy_analyze(strategy)
    for _, val in enumerate(analyze):
        strategy = strategy.copy()
        if val != label and is_ori_label is False:
            strategy['analyze'] = [val, label]
        else:
            strategy['analyze'] = [val]
        result.append(strategy)
    return result


"""
######## Algorithm Setup #######
"""


def algorithm_setup_da_rnn(args, market):
    from algorithm.SL.DualAttnRNN import Algorithm

    # we may need a more precise model_name later
    model_name = "-".join([os.path.basename(__file__).split('.')[0],
                          algorithm_setup_da_rnn.__name__,
                          str(get_timestamp())])

    return Algorithm(tf.Session(config=config), market, market.seq_length, market.data_dim, market.code_count, **{
        "mode": args.mode,
        "hidden_size": 12,
        # "learning_rate": 0.001,
        "layer_size": 2,
        # "keep_prob": 0.98,
        "enable_saver": True,
        "train_steps": 2000,
        "enable_summary_writer": True,
        "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, "stock", "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, "stock", "summary"),
    })


"""
######## Algorithm Setup #######
"""

def bt_datafeed_setup():
    from back_testing.data_mining.strategy import PandasData
    return PandasData


"""
######## Algorithm Setup #######
"""

def bt_strategy_setup():
    from back_testing.OTr_back_test import MyStrategy
    return MyStrategy


class Automation:
    def __init__(self, input_strategy_setup, input_strategy, market_setup, algorithm_setup, bt_strategy_setup, bt_datafeed_setup):
        self._args = model_launcher_parser.parse_args()
        self._input_strategy_origin = input_strategy
        self._input_strategy_post = input_strategy_setup(input_strategy)
        self._market_setup = market_setup
        self._algorithm_setup = algorithm_setup
        self._bt_datafeed_setup = bt_datafeed_setup
        self._bt_datafeed = bt_datafeed_setup()
        self._bt_strategy_setup = bt_strategy_setup
        self._bt_strategy = bt_strategy_setup()


    def fit(self, algorithm):
        algorithm.run()
        return algorithm.eval_and_plot()


    def process(self):
        input_strategies = self._input_strategy_post
        df_output = {}
        for idx, strategy in enumerate(input_strategies):
            market = self._market_setup(self._args, strategy)

            key = next(iter(market.scaled_frames))
            columns = list()
            columns.append(key)
            columns.extend(market.scaled_frames[key].columns.tolist())
            key = "_".join(columns)

            algorithm = self._algorithm_setup(self._args, market)
            result = self.fit(algorithm)
            # get the columns from scaled_frames as the key for the output

            df_output[key] = result

            algorithm.close_sesstion()
            # clear all the graph
            tf.reset_default_graph()
            # generate back testing summary report
        back_testing = BackTesting(MyStrategy=self._bt_strategy,
                                   Datafeed=self._bt_datafeed,
                                   input_dict=df_output,
                                   domain_name=self._input_strategy_origin["name"],
                                   save_input_dict=True
                                   )
        back_testing.backtest()

