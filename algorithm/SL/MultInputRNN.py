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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.contrib.rnn import BasicLSTMCell, RNNCell, MultiRNNCell

from tensorflow.python.ops import array_ops, nn_ops, init_ops
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import math_ops

_BIAS_VARIABLE_NAME = "mi_bias"
_WEIGHTS_VARIABLE_NAME = "mi_kernel"

_BIAS_ATTN = "bias_attention"
_WEIGHT_ATTN = "weight_attention"

_WEIGHT_FINAL_ATTN = "weight_final_attn"
_BIAS_FINAL_ATTN = "bias_final_attn"
_VECTOR_FINAL_ATTN = "vector_final_attn"

_WEIGHT_ACTIVATION = "weight_activation"
_BIAS_ACTIVATION = "bias_activation"

# get all the math operations that will be used
sigmoid = math_ops.sigmoid
relu = tf.nn.relu
multiply = math_ops.multiply
matmul = math_ops.matmul
# add = math_ops.add
tanh = math_ops.tanh
concat = array_ops.concat
split = array_ops.split


class Algorithm(BaseSLTFModel):

    """
    The notation of the comments in the class
        q is the dimensions of LSTM layer
        p is the number of the MI-LSTM hidden units
    """

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

        # tf.Variable(10, name=_WEIGHT_FINAL_ATTN)
        self._w_f_attn = tf.get_variable(_WEIGHT_FINAL_ATTN, [self.hidden_size, self.hidden_size], initializer=self.init)
        self._b_f_attn = tf.get_variable(_BIAS_FINAL_ATTN, [self.hidden_size], initializer=self.init)
        self._v_f_attn = tf.get_variable(_VECTOR_FINAL_ATTN, [self.hidden_size], initializer=self.init)

        self._w_activation = tf.get_variable(_WEIGHT_ACTIVATION, [self.hidden_size], initializer=self.init)
        self._b_activation = tf.get_variable(_BIAS_ACTIVATION, [self.hidden_size], initializer=self.init)

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self):

        self._factor_price, self._factor_pos, self._factor_neg, self._factor_idx = self._split_input()

        self.x = tf.placeholder(tf.float32, [None, self.seq_length, self.x_space])
        self.label = tf.placeholder(tf.float32, [None, self.y_space])


    def _init_nn(self):

        reduce_mean = tf.math.reduce_mean

        def split_ouput(ori_output):
            # TODO
            return None

        # First Attn
        with tf.variable_scope("LSTM_layer"):
            # # self price
            # cell_factor_price = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            # self.out_factor_price, _ = tf.nn.dynamic_rnn(cell_factor_price, self._factor_price, dtype=tf.float32)
            #
            # # positive factors
            # cell_factor_pos = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            # self.out_factor_pos, _ = tf.nn.dynamic_rnn(cell_factor_pos, self._factor_pos, dtype=tf.float32)
            #
            # # negative factors
            # cell_factor_neg = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            # self.out_factor_price, _ = tf.nn.dynamic_rnn(cell_factor_neg, self._factor_neg, dtype=tf.float32)
            #
            # # index
            # cell_factor_index = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)
            # self.out_factor_price, _ = tf.nn.dynamic_rnn(cell_factor_index, self._factor_idx, dtype=tf.float32)

            rnn_cells = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob)

            # out_all, shape = (batch_size, seq_length, cell.output_size)
            out_all, _ = tf.nn.dynamic_rnn(rnn_cells, self.x, dtype=tf.float32)

            # each of the following, shape = (batch_size, seq_length, dimensions)
            # for price and index, the dimension is the hidden_size
            # for positive and negative factors, the dimensions are (P/N, hidden_size)
            out_price, out_pos, out_neg, out_index = self.split_ouput(out_all)

            # we do the average of the positive and negative factors
            out_pos_avg = reduce_mean(out_pos)  # shape = (batch_size, seq_length, q)
            out_neg_avg = reduce_mean(out_neg)  # shape = (batch_size, seq_length, q)

        with tf.variable_scope("MI_RNN_layer"):
            mi_rnn_input = concat([out_price, out_pos_avg, out_neg_avg, out_index], 2) #TODO check ths behavior

            rnn_mi_cells = self.add_rnn(self.layer_size, self.hidden_size, self.keep_prob, MultiInputLSTMCell)

            # out_mi, shape = (batch_size, seq_length, p)
            out_mi, _ = tf.nn.dynamic_rnn(rnn_mi_cells, mi_rnn_input, dtype=tf.float32)

        with tf.variable_scope("final_attn"):
            j = matmul(a=self._w_f_attn,b=out_mi,transpose_b=True)
            j = matmul(a=self._v_f_attn,b=j,transpose_a=True)

            beta = tf.nn.softmax(j, axis=1)
            y_tilde = matmul(a=beta, b=out_mi, transpose_a=True)

        with tf.variable_scope("activation"):

            act_input = nn_ops.bias_add(matmul(self._w_activation,y_tilde), self._b_activation)
            self.y = relu(act_input)



    def _init_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.y, self.label)
        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def train(self):
        for step in range(self.train_steps):
            batch_x, batch_y = self.env.get_batch_data(self.batch_size)
            _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.x: batch_x, self.label: batch_y})
            if (step + 1) % 1000 == 0:
                logging.warning("Step: {0} | Loss: {1:.7f}".format(step + 1, loss))
            if step > 0 and (step + 1) % self.save_step == 0:
                if self.enable_saver:
                    self.save(step)

    def predict(self, x):
        return self.session.run(self.y, feed_dict={self.x: x})


class MultiInputLSTMCell(BasicLSTMCell):
    """
    The notation of the comments in the class
        p is the number of the MI-LSTM hidden units
        q is the input dimensions of MI-LSTM
    """

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 input_divider=None):

        if input_divider is None:
            self._input_divider = 4

        super(MultiInputLSTMCell, self).__init__(num_units=num_units,
                                                 forget_bias=forget_bias,
                                                 state_is_tuple=state_is_tuple,
                                                 activation=activation,
                                                 reuse=reuse,
                                                 name=name,
                                                 dtype=dtype)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value / self._input_divider
        h_depth = self._num_units

        # compared to original LSTM, we have 10 sets of training variable as following than 4:
        # W_f   _kernel[0]
        # W_c, W_cp, W_cn, W_ci _kernel[1:5]
        # W_i, W_ip, W_in, W_ii _kernel[5:9]
        # W_o   _kernel[9]

        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 10 * h_depth])  # [p+q, 10*p]

        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[10 * h_depth],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self._w_attn = self.add_variable(
            _WEIGHT_ATTN,
            shape=[h_depth, h_depth])   # (p, p)

        self._b_attn = self.add_variable(
            _BIAS_ATTN,
            shape=[self._input_divider ],    # for every input piece, we need a bias_alpha
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Multiple Input LSTM
            Args:
              inputs: `2-D` tensor with shape `[batch_size, input_size]`.
              state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size, num_units]`, if `state_is_tuple` has been set to
                `True`.  Otherwise, a `Tensor` shaped
                `[batch_size, 2 * num_units]`.

            Returns:
              A pair containing the new hidden state, and the new state (either a
                `LSTMStateTuple` or a concatenated state, depending on
                `state_is_tuple`).
            """

        def calc_cell_state_tilde(input, h, w, b):
            """
            :param input: shape (B, q)
            :param h: shape (B, p)
            :param w: shape ((p+q), p)
            :param b: shape (p,)    TODO check
            :return: shape (B, p)
            """
            C_t = matmul(
                concat([input, h], 1), w)  # [B, (p+q)] * [(p+q), p] = B * p
            C_t = nn_ops.bias_add(C_t, b)
            return tanh(C_t)

        def calc_input_gate(input, h, w, b):
            """
            :param input: shape (B, q)
            :param h: shape (B, p)
            :param w: shape ((p+q), p)
            :param b: shape (p,)
            :return: shape (B, p)
            """
            input_gate = matmul(
                concat([input, h], 1), w)  # (B, (p+q)) * ((p+q), p) = (B, p)
            input_gate = nn_ops.bias_add(input_gate, b)
            return sigmoid(input_gate)

        def calc_pre_attention(l, w_attn, pre_cell_state, b_attn):
            u = matmul(
                a=l, b=w_attn)  # (B,p) * (p,p) = (B,p)
            u = multiply(   # TODO, check is here correct? element
                u, pre_cell_state)  # (B,p) * (B, p) = (B, p)
            u = tf.reduce_sum(u, 1)
            u = tf.reshape(u, [u.shape[0],1])
            u = nn_ops.bias_add(u, b_attn)
            return tanh(u)

        one = constant_op.constant(1, dtype=dtypes.int32)
        zero = constant_op.constant(0, dtype=dtypes.int32)

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = split(value=state, num_or_size_splits=2, axis=one)

        # TODO check
        """
        Thus matrix W cp , W cn , W ci and biases b cp , b cn , b ci are all initialized to 0 which means that the 
        auxiliary factors are ignored in the very beginning. Hopefully the information from auxiliary factors will 
        gradually ﬂow in with the training process under the control of mainstream.
        """
        W_f, W_c, W_cp, W_cn, W_ci, W_i, W_ip, W_in, W_ii, W_o \
            = split(value=self._kernel, num_or_size_splits=10, axis=one)   # ((p+q), p)

        b_f, b_c, b_cp, b_cn, b_ci, b_i, b_ip, b_in, b_ii, b_o \
            = split(value=self._bias, num_or_size_splits=10, axis=zero)  # (1, p)

        # split the inputs into multiple pieces
        pieces = self._input_divider
        input_pieces = split(value=inputs, num_or_size_splits=pieces, axis=one)

        input_y = input_pieces[0]
        input_p = input_pieces[1]
        input_n = input_pieces[2]
        input_i = input_pieces[3]

        C_tilde_t = calc_cell_state_tilde(input_y, h, W_c, b_c)  # shape = (B,p)
        C_tilde_pt = calc_cell_state_tilde(input_p, h, W_cp, b_cp)
        C_tilde_nt = calc_cell_state_tilde(input_n, h, W_cn, b_cn)
        C_tilde_it = calc_cell_state_tilde(input_i, h, W_ci, b_ci)

        i_t = calc_input_gate(input_y, h, W_i, b_i)  # shape = (B,p)
        i_pt = calc_input_gate(input_y, h, W_ip, b_ip)
        i_nt = calc_input_gate(input_y, h, W_in, b_in)
        i_it = calc_input_gate(input_y, h, W_ii, b_ii)

        l_t = multiply(C_tilde_t, i_t)   # shape = (B,p)
        l_pt = multiply(C_tilde_pt, i_pt)
        l_nt = multiply(C_tilde_nt, i_nt)
        l_it = multiply(C_tilde_it, i_it)

        # get the attention weights and bias
        w_attn = self._w_attn   # shape = (p,p)
        # b_attn =
        b_attn_t, b_attn_pt, b_ttn_nt, b_attn_it, \
            = split(value=self._b_attn, num_or_size_splits=self._input_divider, axis=zero)

        u_t = calc_pre_attention(l_t, w_attn, c, b_attn_t) # shape = (B,1)
        u_pt = calc_pre_attention(l_pt, w_attn, c, b_attn_pt)
        u_nt = calc_pre_attention(l_nt, w_attn, c, b_ttn_nt)
        u_it = calc_pre_attention(l_it, w_attn, c, b_attn_it)

        attn = tf.nn.softmax(concat([u_t, u_pt, u_nt, u_it], axis=1))  # shape of logits: (B, 4)

        attn_t, attn_pt, attn_nt, attn_it = split(value=attn, num_or_size_splits=4, axis=one)  # shape = (B, 1)

        # the ﬁnal cell state input l, shape = (B,p)
        # TODO check the multiply behavior
        l = multiply(l_t, attn_t) + multiply(l_pt, attn_pt) + multiply(l_nt, attn_nt) + multiply(l_it, attn_it)

        # The forget gate and output gate of LSTM remain the same compared with the original LSTM
        # shapes --
        #   input_y: (B, q)
        #   h: (B, p)
        #   W_f: ((p+q), p)
        #   b_f: (p, p)
        f_t = calc_input_gate(input_y, h, W_f, b_f)  # shape (B, p)
        o_t = calc_input_gate(input_y, h, W_o, b_o)

        new_c = multiply(c, f_t) + l  # shape of c and new_c: (B,p)
        new_h = multiply(tanh(new_c), o_t)  # shape new_h: (B,p)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = concat([new_c, new_h], 1)

        return new_h, new_state


def main(args):
    mode = args.mode
    # mode = "test"
    codes = ["nasdaq"]
    # codes = ["600036", "601998"]
    # codes = args.codes
    # codes = ["AU88", "RB88", "CU88", "AL88"]
    market = args.market
    # train_steps = args.train_steps
    # train_steps = 5000
    train_steps = 30000
    # training_data_ratio = 0.98
    training_data_ratio = args.training_data_ratio

    env = Market(codes, start_date="2008-01-01", end_date="2019-02-01", **{
        "market": market,
        "use_sequence": True,
        "scaler": MinMaxScaler(feature_range=(0, 1)),
        "mix_index_state": True,
        "training_data_ratio": training_data_ratio,
    })

    model_name = os.path.basename(__file__).split('.')[0]

    algorithm = Algorithm(tf.Session(config=config), env, env.seq_length, env.data_dim, env.code_count, **{
        "mode": mode,
        "layer_size": 1,
        "hidden_size": 32,
        "keep_prob": 1,   # drop out size = 1 - keep_prob
        "enable_saver": True,
        "train_steps": train_steps,
        "enable_summary_writer": True,
        "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
        "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
    })

    algorithm.run()
    algorithm.eval_and_plot()


if __name__ == '__main__':
    main(model_launcher_parser.parse_args())
