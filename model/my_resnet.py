from tensorflow.contrib import slim
import tensorflow as tf
from hyper_parameter import *


def inference(inputs, num_classes, n):
    """
        total layers 6n+2
    :param inputs:
    :param num_classes:
    :param n:
    :return:
    """
    with slim.arg_scope(arg_scope()):
        net = slim.batch_norm(inputs)
        net = slim.conv2d(net, 16, [3, 3])
        with tf.variable_scope('residual_bolck1'):
            for i in range(n):
                with tf.variable_scope('residual_bolck1_%d' % i):
                    res = net
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 16, [3, 3])
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 16, [3, 3])
                    net = net + res

        with tf.variable_scope('residual_bolck2'):
            for i in range(n):
                with tf.name_scope('residual_bolck2_%d' % i):
                    res = net
                    net = slim.batch_norm(net)
                    if i == 0:
                        net = slim.conv2d(net, 32, [3, 3], stride=2)
                    else:
                        net = slim.conv2d(net, 32, [3, 3])
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 32, [3, 3])
                    if i == 0:
                        res = slim.avg_pool2d(res, [2, 2])
                        net = net + tf.pad(res, [[0, 0], [0, 0], [0, 0], [8, 8]])
                    else:
                        net = net + res

        with tf.variable_scope('residual_bolck3'):
            for i in range(n):
                with tf.variable_scope('residual_bolck3_%d' % i):
                    res = net
                    net = slim.batch_norm(net)
                    if i == 0:
                        net = slim.conv2d(net, 64, [3, 3], stride=2)
                    else:
                        net = slim.conv2d(net, 64, [3, 3])
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, 64, [3, 3])
                    if i == 0:
                        res = slim.avg_pool2d(res, [2, 2])
                        net = net + tf.pad(res, [[0, 0], [0, 0], [0, 0], [16, 16]])
                    else:
                        net = net + res
                    net = slim.batch_norm(net)
        assert net.get_shape().as_list()[1:] == [8, 8, 64]
        with tf.variable_scope('fully_connected'):
            net = tf.reduce_mean(net, [1, 2])
            net = slim.flatten(net)
            logits = slim.fully_connected(net, num_classes)
            # net = slim.dropout(net, keep_prob=0.9, scope='dropout')
        return logits


def arg_scope():
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=None,
                        padding='SAME'):
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            activation_fn=None):
            with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.avg_pool2d], padding='SAME') as sc:
                    return sc
