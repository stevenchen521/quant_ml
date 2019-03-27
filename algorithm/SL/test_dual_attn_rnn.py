from unittest import TestCase
import tensorflow as tf
import os

from algorithm import config
from base.env.market import Market
from checkpoints import CHECKPOINTS_DIR
from sklearn.preprocessing import MinMaxScaler
from algorithm.SL.DualAttnRNN import Algorithm
from helper.args_parser import model_launcher_parser


class TestDualAttnRnn(TestCase):

    def setUp(self):
        self.args = model_launcher_parser.parse_args()

    def test_nasdaq(self):
        mode = self.args.mode
        # mode = "test"
        codes = ["nasdaq"]
        market = self.args.market
        # train_steps = args.train_steps
        # train_steps = 5000
        train_steps = 30000
        # training_data_ratio = 0.98
        training_data_ratio = self.args.training_data_ratio

        env = Market(codes, start_date="2008-01-02", end_date="2019-02-01", **{
            "market": market,
            "use_sequence": True,
            "scaler": MinMaxScaler(feature_range=(0, 1)),
            "mix_index_state": True,
            "training_data_ratio": training_data_ratio,
        })

        model_name = os.path.basename(__file__).split('.')[0]

        algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
            "mode": mode,
            "hidden_size": 48,
            "learning_rate": 0.0001,
            "enable_saver": True,
            "train_steps": train_steps,
            "enable_summary_writer": True,
            "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
        })

        algorithm.run()
        algorithm.eval_and_plot_nasdaq_backtest()

    def test_sh_index(self):
        mode = self.args.mode
        # mode = "test"
        codes = ["SH_index"]
        market = self.args.market
        # train_steps = args.train_steps
        # train_steps = 5000
        train_steps = 30000

        # training_data_ratio = self.args.training_data_ratio
        training_data_ratio = 0.8

        env = Market(codes, start_date="2001-1-02", end_date="2019-02-27", **{
        # env = Market(codes, start_date="2006-10-09", end_date="2019-02-27", **{
            "market": market,
            "use_sequence": True,
            "seq_length": 5,
            "scaler": MinMaxScaler(feature_range=(0, 1)),
            "mix_index_state": True,
            "training_data_ratio": training_data_ratio,
        })

        model_name = os.path.basename(__file__).split('.')[0]
        print(os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"))
        print(os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"))

        algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
            "mode": mode,
            "hidden_size": 256,
            # "learning_rate": 0.001,
            "layer_size": 128,
            # "keep_prob": 0.98,
            "enable_saver": True,
            "train_steps": train_steps,
            "enable_summary_writer": True,
            "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
        })

        algorithm.run()
        algorithm.eval_and_plot_backtest(code=codes[0], model_name=model_name)



