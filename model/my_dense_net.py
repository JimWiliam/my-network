import tensorflow as tf
from tensorflow.contrib import slim


def inference(inputs, num_classes, n, k):
    """
    see https://arxiv.org/pdf/1608.06993.pdf
    total layers 3n+6
    :param inputs:
    :param num_classes:
    :param n: numbers of layers in a dense block
    :param k: growth rate.
    :return: logits
    """
    net = slim.conv2d(inputs, 16, [3, 3], )

    with tf.variable_scope('dense_block_1'):
        for i in range(n):
            with tf.variable_scope('layer%d' % i):
                net_a = slim.batch_norm(net, activation_fn=tf.nn.relu)
                net_a = slim.conv2d(net_a, k, [3, 3], activation_fn=None, )
                net = tf.concat([net_a, net], axis=3)

    with tf.variable_scope('transition1'):
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        _, _, _, c = net.shape
        print(net.shape)
        net = slim.conv2d(net, c, [1, 1])
        net = slim.avg_pool2d(net, [2, 2])

    with tf.variable_scope('dencse_block_2'):
        for i in range(n):
            with tf.variable_scope('layer%d' % i):
                net_a = slim.batch_norm(net, activation_fn=tf.nn.relu)
                net_a = slim.conv2d(net_a, k, [3, 3], activation_fn=None, )
                net = tf.concat([net_a, net], axis=3)

    with tf.variable_scope('transition2'):
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        _, _, _, c = net.shape

        net = slim.conv2d(net, c, [1, 1])
        net = slim.avg_pool2d(net, [2, 2])

    with tf.variable_scope('dencse_block_3'):
        for i in range(n):
            with tf.variable_scope('layer%d' % i):
                net_a = slim.batch_norm(net, activation_fn=tf.nn.relu)
                net_a = slim.conv2d(net_a, k, [3, 3], activation_fn=None)
                net = tf.concat([net_a, net], axis=3)

    with tf.variable_scope('output'):
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = tf.reduce_mean(net, [1, 2])
        net = slim.flatten(net)
        logits = slim.fully_connected(net, num_outputs=num_classes, activation_fn=None)
    return logits
