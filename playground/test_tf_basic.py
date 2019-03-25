import unittest
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops


class MyTestCase(unittest.TestCase):


    CHECKPOINT_DIR = './logs'

    def test_with_session(self):

        with tf.Session() as sess:
            c_0 = tf.constant(42.0, name="c")
            print(c_0.graph)
            print(sess.graph)

    def test_naming_operations(self):
        c_0 = tf.constant(0, name="c")  # => operation named "c"
        # Already-used names will be "uniquified".
        c_1 = tf.constant(2, name="c")  # => operation named "c_1"
        # Name scopes add a prefix to all operations created in the same context.
        with tf.variable_scope("outer"):
            c_2 = tf.constant(2, name="c")  # => operation named "outer/c"
            # Name scopes nest like paths in a hierarchical file system.
            with tf.variable_scope("inner"):
                c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"
            # Exiting a name scope context will return to the previous prefix.
            c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"
            # Already-used name scopes will be "uniquified".
            with tf.variable_scope("inner"):
                c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"

        # with tf.Session() as sess:
            # sess.run(tf.initialize_all_variables())
            # sess.run(tf.global_variables_initializer())
        print(tf.Graph.as_default(self))
        print(c_0.graph)
        print(c_1.graph)
        print(c_2.name)
        print(c_3.name)
        print(c_4.name)
        print(c_5.name)

    def test_graph(self):

        c = tf.constant(value=1)
        # print(assert c.graph is tf.get_default_graph())
        print("c.graph:", c.graph)
        print("tf.get_default_graph:", tf.get_default_graph())

        g = tf.Graph()
        print("g:", g)
        with g.as_default():
            d = tf.constant(value=2)
            print("d.graph:", d.graph)
            # print(g)

        g2 = tf.Graph()
        print("g2:", g2)
        g2.as_default()
        e = tf.constant(value=15)
        print("e.graph", e.graph)


    def test_saver(self):

        x = tf.placeholder(tf.float32, shape=[None, 1])
        y = 4 * x + 4

        w = tf.Variable(tf.random_normal([1], -1, 1))
        b = tf.Variable(tf.zeros([1]))
        y_predict = w * x + b

        loss = tf.reduce_mean(tf.square(y - y_predict))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)

        isTrain = True
        train_steps = 100
        checkpoint_steps = 50
        module_name = 'model.ckpt'

        saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b
        x_data = np.reshape(np.random.rand(10).astype(np.float32), (10, 1))

        tf.summary.FileWriter(os.path.join(self.CHECKPOINT_DIR, module_name)).add_graph(w.graph)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, os.path.join(self.CHECKPOINT_DIR, module_name))

        # with tf.Session() as sess:
        #
        #     sess.run(tf.global_variables_initializer())
        #     if isTrain:
        #         for i in range(train_steps):
        #             sess.run(train, feed_dict={x: x_data})
        #             if (i + 1) % checkpoint_steps == 0:
        #                 saver.save(sess, os.path.join(checkpoint_dir, module_name), global_step=i + 1)
        #     else:
        #         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #         if ckpt and ckpt.model_checkpoint_path:
        #             saver.restore(sess, ckpt.model_checkpoint_path)
        #         else:
        #             pass
        #         print(sess.run(w))
        #         print(sess.run(b))

    def test_tensorboard(self):

        module_name = 'test_variable'
        g_module_name = 'g_test_variable'
        writer = tf.summary.FileWriter(os.path.join(self.CHECKPOINT_DIR, module_name))
        g_writer = tf.summary.FileWriter(os.path.join(self.CHECKPOINT_DIR, g_module_name))

        v_0 = tf.Variable(0, name="v0")  # => operation named "c"


        g = tf.Graph()
        with g.as_default():
            v_1 = tf.Variable(2, name="v1")  # => operation named "outer/c"

          # defaults to saving all variables - in this case w and b


        with tf.Session() as sess:
            saver = tf.train.Saver()
            writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            saver.save(sess, os.path.join(self.CHECKPOINT_DIR, module_name))

        with tf.Session(graph=g) as g_sess:
            g_saver = tf.train.Saver()
            g_writer.add_graph(g_sess.graph)

            g_sess.run(tf.global_variables_initializer())
            g_saver.save(g_sess, os.path.join(self.CHECKPOINT_DIR, g_module_name))

    def test_tensor(self):

        c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        sess = tf.Session()

        print(c.eval(session=sess))

        with sess.as_default():
            split_c0, split_c1 = tf.split(c, [1,2], 1)
            print("split_c0={}".format(split_c0.eval()))
            print("split_c1={}".format(split_c1.eval()))

        split_r0, split_r1 = tf.split(c, [1, 2], 0)
        print("split_r0={}".format(sess.run(split_r0)))
        print("split_r1={}".format(sess.run(split_r1)))


    def test_tensor(self):

        c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        sess = tf.Session()
        one = constant_op.constant(1, dtype=dtypes.int32)

        with sess.as_default():
            # split_even_0,split_even_1,split_even_2 = array_ops.split(value=c, num_or_size_splits=3, axis=one)
            #
            # print("split_even_0={}".format(split_even_0.eval() ))
            # print("split_even_1={}".format(split_even_1.eval() ))
            # print("split_even_2={}".format(split_even_2.eval() ))
            #
            # split_0, split_1 = array_ops.split(value=c, num_or_size_splits=[1,2], axis=one)
            #
            # print("split_0={}".format(split_0.eval() ))
            # print("split_1={}".format(split_1.eval() ))

            split_all = array_ops.split(value=c, num_or_size_splits=[1, 2], axis=one)
            # print("split_all={}".format(split_all.eval()))
            for split_element in split_all:
                print(split_element.eval())

            # print(c.shape[0])
            print("c[0:2,:]={}".format(c[0:2,:].eval()))

    def test_ops(self):
        multiply = math_ops.multiply
        c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a = tf.constant([[1.0], [2.0], [3.0]])

        with tf.Session().as_default():
            result = multiply(c, a)

            # print(a.shape)
            # print(result.eval())

            reduce_mean_res = tf.reduce_mean(c,1)
            print(reduce_mean_res.eval())


    def test_tensor(self):

        # c shape: (32, 10, 8)
        c = tf.constant([[[1.15 for _ in range(8)] for j in range(10)] for i in range(32)])

        # split_c0, split_c1, split_c2, split_c3 = tf.split(c, [1, 3, 3, 1], 2)
        #
        # print(split_c0.shape)
        # print(split_c1.shape)
        # print(split_c2.shape)
        # print(split_c3.shape)

        split_c = tf.split(c, [1, 3, 3, 1], 2)


        for s in split_c:
            split_s = tf.split(s, tf.ones(s.shape[2], dtype='int32'),2)
            # print(type(split_s)) # 'list'
            for i, s2 in enumerate(split_s):
                print(i, s2.shape)

    def test_dimensions(self):
        multiply = math_ops.multiply
        c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a = tf.constant([[1.0], [2.0], [3.0]])

        with tf.Session().as_default():
            result_c = tf.expand_dims(c,0)

            # print(result_c.eval())

            print(tf.concat([result_c,result_c],0).eval())

            tf.reshape(c, [-1, m])



if __name__ == '__main__':
    unittest.main()
