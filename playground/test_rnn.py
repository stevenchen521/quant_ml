import unittest

import tensorflow as tf
from .rnn_enh import MultiInputLSTMCell
from tensorflow.python.framework.tensor_shape import TensorShape
import numpy as np
import os
from helper.util import get_logger


class RNNTestCase(unittest.TestCase):

    LOG_DIR = './logs'


    def setUp(self):
        self.LOGGER = get_logger(__name__)



    def test_basic_rnn_cell(self):
        self.LOGGER.info(self._testMethodName)

        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)  # state_size = 128
        self.LOGGER.info(cell.state_size)  # 128

        inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32 是 batch_size

        h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
        output, h1 = cell.__call__(inputs, h0)  # 调用call函数

        self.LOGGER.info(h1.shape)  # (32, 128)


    def test_basic_rnn_cell2(self):
        self.LOGGER.info(self._testMethodName)

        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128, dtype='float32') # state_size = 128
        self.LOGGER.info(cell.state_size) # 128

        inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
        cell.build(inputs_shape=inputs.shape)

        h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
        output, h1 = cell.call(inputs, h0) #调用cal

        self.LOGGER.info(h1.shape)# (32, 128)

    def test_basic_lstm_cell(self):
        self.LOGGER.info(self._testMethodName)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
        inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32 是 batch_size
        h0 = lstm_cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
        output, h1 = lstm_cell.__call__(inputs, h0)

        self.LOGGER.info(h1.h)  # shape=(32, 128)
        self.LOGGER.info(h1.c)  # shape=(32, 128)

    def test_multi_rnn_cell(self):
        self.LOGGER.info(self._testMethodName)

        # 每调用一次这个函数就返回一个BasicRNNCell
        def get_a_cell():
            return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

        # 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])  # 3层RNN
        # 得到的cell实际也是RNNCell的子类
        # 它的state_size是(128, 128, 128)
        # (128, 128, 128)并不是128x128x128的意思
        # 而是表示共有3个隐层状态，每个隐层状态的大小为128
        self.LOGGER.info(cell.state_size)  # (128, 128, 128)
        # 使用对应的call函数
        inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32 是 batch_size
        h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
        output, h1 = cell.call(inputs, h0)
        self.LOGGER.info(h1)  # tuple中含有3个32x128的向量

        # sess = tf.Session()
        # self.LOGGER.info(sess.run())


    def test_dynamic_rnn(self):
        self.LOGGER.info(self._testMethodName)

        # 每调用一次这个函数就返回一个BasicRNNCell
        def get_a_cell():
            return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

        # 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
        cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])  # 3层RNN
        # 得到的cell实际也是RNNCell的子类
        # 它的state_size是(128, 128, 128)
        # (128, 128, 128)并不是128x128x128的意思
        # 而是表示共有3个隐层状态，每个隐层状态的大小为128
        # self.LOGGER.info(cell.state_size)  # (128, 128, 128)
        # 使用对应的call函数
        inputs = tf.placeholder(np.float32, shape=(32,10, 100))  # 32 是 batch_size
        h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态

        outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0)
        tf.global_variables_initializer()
        # sess = tf.Session()
        # self.LOGGER.info(sess.run())

    def test_drnn_with_session(self):
        self.LOGGER.info(self._testMethodName)

        module_name = 'test_drnn_with_session'
        writer = tf.summary.FileWriter(os.path.join(self.LOG_DIR, module_name))


        # 每调用一次这个函数就返回一个BasicRNNCell
        def get_a_cell():
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=128)

        with tf.variable_scope("MultiRNNCell"):
        # 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])  # 3层RNN
            h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
        # 得到的cell实际也是RNNCell的子类
        # 它的state_size是(128, 128, 128)
        # (128, 128, 128)并不是128x128x128的意思
        # 而是表示共有3个隐层状态，每个隐层状态的大小为128
        # self.LOGGER.info(cell.state_size)  # (128, 128, 128)
        # 使用对应的call函数

        with tf.variable_scope("inputs"):
            inputs = tf.placeholder(np.float32, shape=(32, 10, 100))  # 32 是 batch_size


        with tf.variable_scope("dynamic_rnn"):
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0)

        # sess = tf.Session()
        # self.LOGGER.info(sess.run())


        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()
            writer.add_graph(sess.graph)
            tf.global_variables_initializer()

            train_data_feed = {
                inputs: [[[1.15 for k in range(100)] for j in range(10)] for i in range(32)]
            }
            
            a = sess.run([inputs], train_data_feed)
            saver.save(sess, os.path.join(self.LOG_DIR, module_name))

    def test_multi_input_lstm(self):
        self.LOGGER.info(self._testMethodName)

        cell = MultiInputLSTMCell(num_units=128)  # state_size = 128
        # self.LOGGER.info(cell.state_size)  # 128
        cell.build(inputs_shape=TensorShape([32,100]))

        self.LOGGER.info(tf.contrib.framework.get_trainable_variables())

    def test_mirnn_with_session(self):
        self.LOGGER.info(self._testMethodName)

        # module_name = 'test_mirnn_with_session'
        writer = tf.summary.FileWriter(os.path.join(self.LOG_DIR, self._testMethodName))

        # 每调用一次这个函数就返回一个BasicRNNCell
        def get_a_cell():
            # return MultiInputLSTMCell(num_units=128, memory_slots=2, memory_size=128, keys={"key1":1})
            return MultiInputLSTMCell(num_units=128)

        with tf.variable_scope("MultiRNNCell"):
            #
            cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])  # 3层RNN
            h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态

            # cell = get_a_cell()
            # h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
        # 得到的cell实际也是RNNCell的子类
        # 它的state_size是(128, 128, 128)
        # (128, 128, 128)并不是128x128x128的意思
        # 而是表示共有3个隐层状态，每个隐层状态的大小为128
        # self.LOGGER.info(cell.state_size)  # (128, 128, 128)
        # 使用对应的call函数

        with tf.variable_scope("inputs"):
            inputs = tf.placeholder(np.float32, shape=(32, 10, 100))  # 32 是 batch_size

        with tf.variable_scope("dynamic_rnn"):
            # outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=float)


        # sess = tf.Session()
        # self.LOGGER.info(sess.run())

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()
            writer.add_graph(sess.graph)
            tf.global_variables_initializer()

            train_data_feed = {
                inputs: [[[1.15 for k in range(100)] for j in range(10)] for i in range(32)]
            }

            a = sess.run([inputs], train_data_feed)
            saver.save(sess, os.path.join(self.LOG_DIR, self._testMethodName))

            self.LOGGER.info(tf.contrib.framework.get_trainable_variables())


    def test_tensorflow_basic(self):
        self.LOGGER.info(self._testMethodName)

        # Create a tensor
        c = tf.constant([0 for _ in range(128)])
        self.LOGGER.info(c)

        # Expand one dimension
        e = tf.expand_dims(c, 0)
        self.LOGGER.info(e)

        # duplicate 32 rows/batch size
        t = tf.tile(e, [32, 1])
        self.LOGGER.info(t)



if __name__ == '__main__':
    unittest.main()
