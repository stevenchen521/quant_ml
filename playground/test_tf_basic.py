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
        # self.CHECKPOINT_DIR = 'session_graph'
        # print (os.path.join(self.CHECKPOINT_DIR, module_name))
        writer = tf.summary.FileWriter(os.path.join(self.CHECKPOINT_DIR, module_name))
        g_writer = tf.summary.FileWriter(os.path.join(self.CHECKPOINT_DIR, g_module_name))

        v_0 = tf.Variable(0, name="v0")  # => operation named "c"
        g = tf.Graph()
        with g.as_default():
            v_1 = tf.Variable(2, name="v1")  # => operation named "outer/c"

          # defaults to saving all variables - in this case w and b


        with tf.Session() as sess:
            saver = tf.train.Saver()


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
            print("c[0:2,:]={}".format(c[0:2, :].eval()))

    def test_ops(self):
        multiply = math_ops.multiply
        c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a = tf.constant([[1.0], [2.0], [3.0]])

        with tf.Session().as_default():
            result = multiply(c, a)

            # print(a.shape)
            print(result.eval())

            reduce_mean_res = tf.reduce_mean(result, 1)
            print(reduce_mean_res.shape)


    def test_tensor(self):

        # c shape: (32, 10, 8)
        c = tf.constant([[[1.15 for _ in range(8)] for j in range(10)] for i in range(32)])

        split_c0, split_c1, split_c2, split_c3 = tf.split(c, [1, 3, 3, 1], 2)

        print(split_c0.shape)
        print(split_c1.shape)
        print(split_c2.shape)
        print(split_c3.shape)
        # with tf.Session().as_default():
        #     print (c.eval())

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


    def test_tf_hello_world(self):
        import tensorflow as tf
        import numpy as np

        # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
        x_data = np.float32(np.random.rand(2, 100))  # 随机输入
        y_data = np.dot([0.100, 0.200], x_data) + 0.300

        # 构造一个线性模型
        #
        b = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
        y = tf.matmul(W, x_data) + b

        # 最小化方差
        loss = tf.reduce_mean(tf.square(y - y_data))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)

        # 初始化变量
        init = tf.initialize_all_variables()

        # 启动图 (graph)
        sess = tf.Session()
        sess.run(init)

        # 拟合平面
        for step in range(0, 201):
            sess.run(train)
            if step % 20 == 0:
                print(step, sess.run(W), sess.run(b))

        # 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]

    def test_tensor_board2(self):
        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.

            It does a matrix multiply, bias add, and then uses relu to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    tf.summary.histogram('pre_activations', preactivate)
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

        hidden1 = nn_layer(x, 784, 500, 'layer1')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden1, keep_prob)

        # Do not apply softmax activation yet, see below.
        y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

        with tf.name_scope('cross_entropy'):
            # The raw formulation of cross-entropy,
            #
            # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
            #                               reduction_indices=[1]))
            #
            # can be numerically unstable.
            #
            # So here we use tf.losses.sparse_softmax_cross_entropy on the
            # raw logit outputs of the nn_layer above.
            with tf.name_scope('total'):
                cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(0.8).minimize(
                cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
        tf.global_variables_initializer().run()



if __name__ == '__main__':
    unittest.main()
