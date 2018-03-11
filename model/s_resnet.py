from tensorflow.contrib import slim
import tensorflow as tf


def inference(inputs, num_classes, n):
    """
        total layers 6n+2
    :param inputs:
    :param num_classes:
    :param n:
    :return:
    """
    net = slim.conv2d(inputs, 16, [3, 3], 1)
    net = slim.batch_norm(net)

    with tf.variable_scope('residual_bolck1'):
        for i in range(n):
            with tf.variable_scope('residual_bolck1_%d' % i):
                res = net
                net = slim.conv2d(net, 16, [3, 3], activation_fn=None)
                net = slim.batch_norm(net, activation_fn=tf.nn.relu)

                net_a = slim.separable_conv2d(net, 8, [1, 1], depth_multiplier=1, activation_fn=None)
                net_b = slim.separable_conv2d(net, 8, [3, 3], depth_multiplier=1, activation_fn=None)
                net = tf.concat([net_a, net_b], axis=3)

                net = net + res
                net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    with tf.variable_scope('residual_bolck2'):
        for i in range(n):
            with tf.name_scope('residual_bolck2_%d' % i):
                res = net
                if i == 0:
                    net = slim.conv2d(net, 32, [3, 3], stride=2, activation_fn=None)
                else:
                    net = slim.conv2d(net, 32, [3, 3], activation_fn=None)
                net = slim.batch_norm(net, activation_fn=tf.nn.relu)

                net_a = slim.separable_conv2d(net, 16, [1, 1], depth_multiplier=1, activation_fn=None)
                net_b = slim.separable_conv2d(net, 16, [3, 3], depth_multiplier=1, activation_fn=None)
                net = tf.concat([net_a, net_b], axis=3)

                if i == 0:
                    res = slim.avg_pool2d(res, [2, 2])
                    net = net + tf.pad(res, [[0, 0], [0, 0], [0, 0], [8, 8]])
                else:
                    net = net + res
                net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    with tf.variable_scope('residual_bolck3'):
        for i in range(n):
            with tf.variable_scope('residual_bolck3_%d' % i):
                res = net
                if i == 0:
                    net = slim.conv2d(net, 64, [3, 3], stride=2, activation_fn=None)
                else:
                    net = slim.conv2d(net, 64, [3, 3], activation_fn=None)

                net = slim.batch_norm(net, activation_fn=tf.nn.relu)

                net_a = slim.separable_conv2d(net, 32, [1, 1], depth_multiplier=1, activation_fn=None)
                net_b = slim.separable_conv2d(net, 32, [3, 3], depth_multiplier=1, activation_fn=None)
                net = tf.concat([net_a, net_b], axis=3)

                if i == 0:
                    res = slim.avg_pool2d(res, [2, 2])
                    net = net + tf.pad(res, [[0, 0], [0, 0], [0, 0], [16, 16]])
                else:
                    net = net + res
                net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    assert net.get_shape().as_list()[1:] == [8, 8, 64]
    with tf.variable_scope('fully_connected'):
        net = tf.reduce_mean(net, [1, 2])
        assert net.get_shape().as_list()[-1:] == [64]
        net = slim.flatten(net)
        net = slim.fully_connected(net, num_classes, activation_fn=None)
        # net = slim.dropout(net, keep_prob=0.9, scope='dropout')
        net = slim.batch_norm(net)
        logits = slim.softmax(net)
    return logits
