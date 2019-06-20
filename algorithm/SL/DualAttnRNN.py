# coding=utf-8
import tensorflow as tf
import logging
import os
from algorithm import config
from base.env.market import Market
from checkpoints import CHECKPOINTS_DIR
from base.algorithm.model import BaseSLTFModel
from sklearn.preprocessing import MinMaxScaler
from helper.args_parser import model_launcher_parser
from back_testing.localplayground.OTr_back_test import PandasDeepLearning, MyStrategy


class Algorithm(BaseSLTFModel):
    def __init__(self, session, env, seq_length, x_space, y_space, **options):
        super(Algorithm, self).__init__(session, env, **options)

        self.seq_length, self.x_space, self.y_space = seq_length, x_space, y_space

        try:
            self.hidden_size = options['hidden_size']
            self.layer_size = options['layer_size']
            self.keep_prob = options['keep_prob']
        except KeyError:
            self.hidden_size = 1
            self.layer_size = 1
            self.keep_prob = 1

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self):
        self.x = tf.placeholder(tf.float32, [None, self.seq_length, self.x_space])
        self.label = tf.placeholder(tf.float32, [None, self.y_space])

        self.learning_rate_tensor = tf.placeholder(tf.float32, None, name="learning_rate")

    def _init_nn(self):
        # First Attn
        with tf.variable_scope("1st_encoder"):
            self.f_encoder_rnn = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            self.f_encoder_outputs, _ = tf.nn.dynamic_rnn(self.f_encoder_rnn, self.x, dtype=tf.float32)
            self.f_attn_inputs = self.add_fc(self.f_encoder_outputs, self.hidden_size, tf.tanh)
            self.f_attn_outputs = tf.nn.softmax(self.f_attn_inputs)
        with tf.variable_scope("1st_decoder"):
            self.f_decoder_input = tf.multiply(self.f_encoder_outputs, self.f_attn_outputs)
            self.f_decoder_rnn = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            self.f_decoder_outputs, _ = tf.nn.dynamic_rnn(self.f_decoder_rnn, self.f_decoder_input, dtype=tf.float32)
        # Second Attn
        with tf.variable_scope("2nd_encoder"):
            self.s_attn_input = self.add_fc(self.f_decoder_outputs, self.hidden_size, tf.tanh)
            self.s_attn_outputs = tf.nn.softmax(self.s_attn_input)
        with tf.variable_scope("2nd_decoder"):
            self.s_decoder_input = tf.multiply(self.f_decoder_outputs, self.s_attn_outputs)
            self.s_decoder_rnn = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            self.f_decoder_outputs, _ = tf.nn.dynamic_rnn(self.s_decoder_rnn, self.s_decoder_input, dtype=tf.float32)
            self.f_decoder_outputs_dense = self.add_fc(self.f_decoder_outputs[:, -1], 16)
            self.y = self.add_fc(self.f_decoder_outputs_dense, self.y_space)

    def _init_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.y, self.label)
        with tf.variable_scope('loss_test'):
            self.loss_test = tf.losses.mean_squared_error(self.y, self.label)
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)

            # should we use the the decay???
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self):
        for step in range(self.train_steps):
            learning_rate = self.learning_rate * (
                    self.learning_rate_decay ** max(float(step/1000 + 1), 0.0)
            )
            batch_x, batch_y = self.env.get_batch_data(self.batch_size)
            train_data_feed = {
                # self.learning_rate_tensor: learning_rate,
                self.x: batch_x,
                self.label: batch_y
                # empty one dimensional tensor
            }
            _, loss = self.session.run([self.train_op, self.loss], feed_dict=train_data_feed)
            if (step + 1) % 1000 == 0:
                # logging.warning("Step: {0} |Loss: {1:.7f}".format(step + 1, loss))
                test_x, label = self.env.get_test_data()
                test_data_feed = {
                    # self.learning_rate_tensor: 0,
                    self.x: test_x,
                    self.label: label
                }
                test_pred, loss_test = self.session.run([self.y, self.loss_test], feed_dict=test_data_feed)
                logging.warning("Step: {0} |train_loss: {1:.7f} |Test_Loss: {2:.7f}"
                                .format(step + 1, loss, loss_test))
            if step > 0 and (step + 1) % self.save_step == 0:
                if self.enable_saver:
                    self.save(step)

    def predict(self, x):
        return self.session.run(self.y, feed_dict={self.x: x})


def main(args):
    mode = args.mode
    # mode = "test"
    codes = ["SH_index_all"]
    # codes = ["600036", "601998"]
    # codes = args.codes
    # codes = ["AU88", "RB88", "CU88", "AL88"]
    market = args.market
    # train_steps = args.train_steps
    # train_steps = 5000
    train_steps = 30000
    # training_data_ratio = 0.98
    training_data_ratio = args.training_data_ratio

    env = Market(codes, start_date="2001-01-03", end_date="2019-03-08", **{
        "market": market,
        "use_sequence": True,
        "seq_length": 5,
        "scaler": MinMaxScaler(feature_range=(0, 1)),
        "mix_index_state": True,
        "training_data_ratio": 0.8,
    })

    model_name = os.path.basename(__file__).split('.')[0]

    algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
        "mode": mode,
        "layer_size": 2,
        "hidden_size": 48,
        # "keep_prob": 0.98,   # drop out size = 1 - keep_prob
        "enable_saver": True,
        "train_steps": train_steps,
        "enable_summary_writer": True,
        "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
    })

    algorithm.run()
    # algorithm.eval_and_plot_backtest_single(code=codes[0],
    #                                         model_name=model_name,
    #                                         MyStrategy=MyStrategy,
    #                                         DataFeed=PandasDeepLearning,
    #                                         plot=True)
    algorithm.eval_and_plot_backtest(code=codes[0], model_name=model_name)
    # wirter = tf.summary.FileWriter("")



if __name__ == '__main__':
    main(model_launcher_parser.parse_args())
