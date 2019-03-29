import tensorflow as tf
import pandas as pd
import numpy as np
import json

from abc import abstractmethod
from helper import data_ploter
from tensorflow.contrib import rnn
from helper.data_logger import generate_algorithm_logger
import re
from back_testing.common.common import BackTesting
import math
from functools import reduce

def get_project_rootpath(project_name):
    import os
    file_path = os.getcwd()
    my_path = re.findall(r'.*{}'.format(project_name), file_path)[0]
    return my_path


class BaseTFModel(object):

    def __init__(self, session, env, **options):
        self.session = session
        self.env = env
        self.total_step = 0
        try:
            self.learning_rate_decay = options['learning_rate_decay']
        except KeyError:
            self.learning_rate_decay = 0.99

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.logger = options['logger']
        except KeyError:
            self.logger = generate_algorithm_logger('model')

        try:
            self.enable_saver = options["enable_saver"]
        except KeyError:
            self.enable_saver = False

        try:
            self.enable_summary_writer = options['enable_summary_writer']
        except KeyError:
            self.enable_summary_writer = False

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        try:
            self.summary_path = options["summary_path"]
        except KeyError:
            self.summary_path = None

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    def restore(self):
        self.saver.restore(self.session, self.save_path)

    def _init_saver(self):
        if self.enable_saver:
            self.saver = tf.train.Saver()

    def _init_summary_writer(self):
        if self.enable_summary_writer:
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.session.graph)

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        return None, None, None

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def add_rnn(layer_count, hidden_size, keep_prob=1, cell=rnn.BasicLSTMCell, activation=tf.tanh):

        def _create_one_cell(cell, hidden_size, activation, keep_prob):
            rnn_cell = cell(hidden_size, activation=activation)
            if keep_prob != 1:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
            return rnn_cell

        cells = [_create_one_cell(cell, hidden_size, activation, keep_prob) for _ in range(layer_count)]
        return rnn.MultiRNNCell(cells)

    # @staticmethod
    # def add_rnn(layer_count, hidden_size, cell=rnn.BasicLSTMCell, activation=tf.tanh):
    #     cells = [cell(hidden_size, activation=activation) for _ in range(layer_count)]
    #     return rnn.MultiRNNCell(cells)

    def _create_one_cell(cell, hidden_size, activation, keep_prob):
        rnn_cell = cell(hidden_size, activation=activation, output_keep_prob=keep_prob)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
        return rnn_cell

    @staticmethod
    def add_cnn(x_input, filters, kernel_size, pooling_size):
        convoluted_tensor = tf.layers.conv2d(x_input, filters, kernel_size, padding='SAME', activation=tf.nn.relu)
        return tf.layers.max_pooling2d(convoluted_tensor, pooling_size, strides=[1, 1], padding='SAME')

    @staticmethod
    def add_fc(x, units, activation=None):
        return tf.layers.dense(x, units, activation=activation)


class BaseRLTFModel(BaseTFModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(BaseRLTFModel, self).__init__(session, env, **options)

        # Initialize evn parameters.
        self.a_space, self.s_space = a_space, s_space

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 30

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.epsilon = options['epsilon']
        except KeyError:
            self.epsilon = 0.9

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 10000

        try:
            self.save_episode = options["save_episode"]
        except KeyError:
            self.save_episode = 10

    def eval(self):
        self.mode = 'test'
        s = self.env.reset('eval')
        while True:
            c, a, _ = self.predict(s)
            s_next, r, status, info = self.env.forward(c, a)
            s = s_next
            if status == self.env.Done:
                self.env.trader.log_asset(0)
                break

    def plot(self):
        with open(self.save_path + '_history_profits.json', mode='w') as fp:
            json.dump(self.env.trader.history_profits, fp, indent=True)

        with open(self.save_path + '_baseline_profits.json', mode='w') as fp:
            json.dump(self.env.trader.history_baselines, fp, indent=True)

        data_ploter.plot_profits_series(
            self.env.trader.history_baselines,
            self.env.trader.history_profits,
            self.save_path
        )

    def save(self, episode):
        self.saver.save(self.session, self.save_path)
        self.logger.warning("Episode: {} | Saver reach checkpoint.".format(episode))

    @abstractmethod
    def save_transition(self, s, a, r, s_next):
        pass

    @abstractmethod
    def log_loss(self, episode):
        pass

    @staticmethod
    def get_a_indices(a):
        a = np.where(a > 1 / 3, 2, np.where(a < - 1 / 3, 1, 0)).astype(np.int32)[0].tolist()
        return a

    def get_stock_code_and_action(self, a, use_greedy=False, use_prob=False):
        # Reshape a.
        if not use_greedy:
            a = a.reshape((-1,))
            # Calculate action index depends on prob.
            if use_prob:
                # Generate indices.
                a_indices = np.arange(a.shape[0])
                # Get action index.
                action_index = np.random.choice(a_indices, p=a)
            else:
                # Get action index.
                action_index = np.argmax(a)
        else:
            if use_prob:
                # Calculate action index
                if np.random.uniform() < self.epsilon:
                    action_index = np.floor(a).astype(int)
                else:
                    action_index = np.random.randint(0, self.a_space)
            else:
                # Calculate action index
                action_index = np.floor(a).astype(int)

        # Get action
        action = action_index % 3
        # Get stock index
        stock_index = np.floor(action_index / 3).astype(np.int)
        # Get stock code.
        stock_code = self.env.codes[stock_index]

        return stock_code, action, action_index


class BaseSLTFModel(BaseTFModel):

    def __init__(self, session, env, **options):
        super(BaseSLTFModel, self).__init__(session, env, **options)

        # Initialize parameters.
        self.x, self.label, self.y, self.loss = None, None, None, None

        try:
            self.train_steps = options["train_steps"]
        except KeyError:
            self.train_steps = 30000

        try:
            self.save_step = options["save_step"]
        except KeyError:
            self.save_step = 1000

    def run(self):
        if self.mode == 'train':
            self.train()
        else:
            self.restore()

    def save(self, step):
        self.saver.save(self.session, self.save_path)
        self.logger.warning("Step: {} | Saver reach checkpoint.".format(step + 1))

    def eval_and_plot(self):
        x, label = self.env.get_test_data()
        y = self.predict(x)
        with open(self.save_path + '_y.json', mode='w') as fp:
            json.dump(y.tolist(), fp, indent=True)

        with open(self.save_path + '_label.json', mode='w') as fp:
            json.dump(label.tolist(), fp, indent=True)
        data_ploter.plot_stock_series(self.env.codes,
                                      y,
                                      label,
                                      self.save_path)
        date_index = self.env.dates[self.env.e_data_indices[0] + self.env.seq_length + 1:]
        dataframe_backtest = pd.DataFrame({'label': label.flatten(),
                                           'y': y.flatten()}, index=date_index)
        original_frames = self.env.origin_frames[self.env.codes[0]][self.env.e_data_indices[0] + self.env.seq_length:]
        dataframe_backtest = pd.concat([dataframe_backtest, original_frames], axis=1)
        return dataframe_backtest


    # def format_model_output(self, x, label, y):
    #     date_index = self.env.dates[self.env.e_data_indices[0] + self.env.seq_length + 1:]
    #     frames_output = pd.DataFrame({'label': label.flatten(),
    #                                  'y': y.flatten()}, index=date_index)
    #     frames_test = self.env.origin_frames[self.env.codes[0]].loc[date_index]
    #     frames_output = pd.concat([frames_test, frames_output], axis=1)
    #     frames_output['date'] = frames_output.index
    #     frames_output['date'] = frames_output['date'].apply(
    #         lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
    #     # frames_output.dropna(how="any", inplace=True)
    #     # frames_output.set_index(['Date'], inplace=True)
    #     frames_output['openinterest'] = 0
    #     col_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest', 'label', 'y']
    #     frames_output = frames_output[col_order]
    #     return frames_output


    def eval_and_plot_backtest(self, code, model_name):
        x, label = self.env.get_test_data()
        y = self.predict(x)
        with open(self.save_path + '_y.json', mode='w') as fp:
            json.dump(y.tolist(), fp, indent=True)

        with open(self.save_path + '_label.json', mode='w') as fp:
            json.dump(label.tolist(), fp, indent=True)
        data_ploter.plot_stock_series(self.env.codes,
                                      y,
                                      label,
                                      self.save_path)
        # reformat the data and transformed the data to back testing data
        date_index = self.env.dates[self.env.e_data_indices[0] + self.env.seq_length + 1:]
        dataframe_backtest = pd.DataFrame({'label': label.flatten(),
                                           'y': y.flatten()}, index=date_index)
        original_frames = self.env.origin_frames[self.env.codes[0]][self.env.e_data_indices[0] + self.env.seq_length:]
        dataframe_backtest = pd.concat([dataframe_backtest, original_frames], axis=1)
        dataframe_backtest['date'] = dataframe_backtest.index
        dataframe_backtest['date'] = dataframe_backtest['date'].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
        dataframe_backtest.dropna(how="any", inplace=True)
        dataframe_backtest.set_index(['date'], inplace=True)
        dataframe_backtest['openinterest'] = 0
        col_order = ['open', 'high', 'low', 'close', 'volume', 'openinterest', 'label', 'y']
        dataframe_backtest = dataframe_backtest[col_order]
        # dataframe_backtest = dataframe_backtest.round(2)
        dataframe_backtest.to_csv("../../back_testing/data/{}_{}_for_backtest.csv".format(model_name, code), date_format='%Y-%m-%d %H:%M:%S')
        print ('dataframe is updated')


    # @staticmethod
    def format_backtest_frame(self, dataframe_backtest):
        dataframe_backtest['date'] = dataframe_backtest.index
        dataframe_backtest['date'] = dataframe_backtest['date'].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
        dataframe_backtest.index = range(len(dataframe_backtest))
        dataframe_backtest.dropna(how="any", inplace=True)
        # dataframe_backtest.set_index(['date'], inplace=True)
        dataframe_backtest['openinterest'] = 0
        col_order = ['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest', 'label', 'y']
        dataframe_backtest = dataframe_backtest[col_order]
        return dataframe_backtest


    def eval_and_plot_backtest_single(self, code, model_name, DataFeed, MyStrategy, summary_path = False, plot = False):
        df = self.eval_and_plot()
        # reformat the data and transformed the data to back testing data
        dataframe_backtest = self.format_backtest_frame(df)
        mypath = get_project_rootpath('quant_ml')
        if summary_path:
            save_path = summary_path
        else:
            save_path = mypath + "/back_testing/summary_excel/{}_{}_backtest_summary.xlsx".format(model_name, code)
        data_dict = dict()
        dataframe_backtest.index = range(len(dataframe_backtest))
        data_dict[code] = dataframe_backtest
        B = BackTesting(
            input_dict=data_dict,
            MyStrategy=MyStrategy,
            Datafeed=DataFeed,
            domain_name=model_name,
            summary_path=save_path,
            save_input_dict=True,
            plot=plot)
        B.backtest()


    def close_sesstion(self):
        self.session.close()



class BasePTModel(object):

    def __init__(self, env, **options):

        self.env = env

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.save_path = options["save_path"]
        except KeyError:
            self.save_path = None

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, a):
        pass

    @abstractmethod
    def restore(self):
        pass

    @abstractmethod
    def run(self):
        pass


class BaseRLPTModel(BasePTModel):

    def __init__(self, env, a_space, s_space, **options):
        super(BaseRLPTModel, self).__init__(env, **options)

        self.env = env

        self.a_space, self.s_space = a_space, s_space

        try:
            self.episodes = options['episodes']
        except KeyError:
            self.episodes = 30

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 2000

        try:
            self.mode = options['mode']
        except KeyError:
            self.mode = 'train'

    @abstractmethod
    def _init_input(self, *args):
        pass

    @abstractmethod
    def _init_nn(self, *args):
        pass

    @abstractmethod
    def _init_op(self):
        pass

    @abstractmethod
    def save_transition(self, s, a, r, s_n):
        pass

    @abstractmethod
    def log_loss(self, episode):
        pass

    @staticmethod
    def get_a_indices(a):
        a = np.where(a > 1 / 3, 2, np.where(a < - 1 / 3, 1, 0)).astype(np.int32)[0].tolist()
        return a
