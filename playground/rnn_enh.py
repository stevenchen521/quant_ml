import tensorflow as tf

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
            shape=[self._input_divider],    # for every input piece, we need a bias_alpha
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

        # get all the math operations that will be used
        sigmoid = math_ops.sigmoid
        multiply = math_ops.multiply
        matmul = math_ops.matmul
        # add = math_ops.add
        tanh = math_ops.tanh
        concat = array_ops.concat
        split = array_ops.split

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
            # u = tf.reshape(u, [u.shape[0],1])
            u = tf.expand_dims(u, 1)
            u = nn_ops.bias_add(u, b_attn)
            return tanh(u)

        one = constant_op.constant(1, dtype=dtypes.int32)
        zero = constant_op.constant(0, dtype=dtypes.int32)

        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = split(value=state, num_or_size_splits=2, axis=one)

        # W_f = self._kernel[0,:]
        # W_c = self._kernel[1,:] # [(p+q), p]
        # W_cp = self._kernel[2,:] # [(p+q), p]
        # W_cn = self._kernel[3,:] # [(p+q), p]
        # W_ci = self._kernel[4,:] # [(p+q), p]
        # W_i = self._kernel[5,:]
        # W_ip = self._kernel[6,:]
        # W_in = self._kernel[7,:]
        # W_ii = self._kernel[8,:]
        # W_o = self._kernel[9,:]

        # b_f = self._bias[0, :]
        # b_c = self._bias[1, :]  # [(p+q), p]
        # b_cp = self._bias[2, :]  # [(p+q), p]
        # b_cn = self._bias[3, :]  # [(p+q), p]
        # b_ci = self._bias[4, :]  # [(p+q), p]
        # b_i = self._bias[5, :]
        # b_ip = self._bias[6, :]
        # b_in = self._bias[7, :]
        # b_ii = self._bias[8, :]
        # b_o = self._bias[9, :]

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
