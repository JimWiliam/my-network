import tensorflow as tf
from tensorflow.contrib import slim
from tflearn.layers.conv import global_avg_pool


def model(inputs, num_classes):
    with tf.name_scope('stem'):
        net = slim.conv2d(inputs, 32, [3, 3], stride=2)
        net = slim.conv2d(net, 32, [3, 3])
        net = slim.conv2d(net, 64, [3, 3])
        net_a = slim.max_pool2d(net, [2, 2])
        net_b = slim.conv2d(net, 96, [3, 3], stride=2)

        net = tf.concat([net_a, net_b], axis=3)
        net_a = slim.conv2d(net, 64, [1, 1], )
        net_a = slim.conv2d(net_a, 96, [3, 3])
        net_b = slim.conv2d(net, 64, [1, 1])
        net_b = slim.conv2d(net_b, 64, [7, 1], )
        net_b = slim.conv2d(net_b, 64, [1, 7])
        net_b = slim.conv2d(net_b, 96, [3, 3])

        net = tf.concat([net_a, net_b], axis=3)
        net_a = slim.conv2d(net, 192, [3, 3], stride=2)
        net_b = slim.max_pool2d(net, [2, 2], stride=2)
        net = tf.concat([net_a, net_b], axis=3)
        net = tf.nn.relu(net)

    for i in range(5):
        with tf.name_scope('inception_resnet_a_%d' % i):
            x = slim.conv2d(net, 384, [1, 1])

            net_a = slim.conv2d(net, 32, [1, 1])

            net_b = slim.conv2d(net, 32, [1, 1])
            net_b = slim.conv2d(net_b, 32, [3, 3])

            net_c = slim.conv2d(net, 32, [1, 1])
            net_c = slim.conv2d(net_c, 48, [3, 3])
            net_c = slim.conv2d(net_c, 64, [3, 3])

            net = tf.concat([net_a, net_b, net_c], axis=3)
            net = slim.conv2d(net, 384, [1, 1], activation_fn=None)

            net = x + net
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    with tf.name_scope('reduction_a'):
        k = 256
        l = 256
        m = 384
        n = 384
        net_a = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME')

        net_b = slim.conv2d(net, n, [3, 3], stride=2)

        net_c = slim.conv2d(net, k, [1, 1])
        net_c = slim.conv2d(net_c, l, [3, 3])
        net_c = slim.conv2d(net_c, m, [3, 3], stride=2)

        net = tf.concat([net_a, net_b, net_c], axis=3)
        net = tf.nn.relu(net)

    for i in range(10):
        with tf.name_scope('inception_resnet_b_%d' % i):
            x = slim.conv2d(net, 1154, [1, 1])

            net_a = slim.conv2d(net, 192, [1, 1])

            net_b = slim.conv2d(net, 128, [1, 1])
            net_b = slim.conv2d(net_b, 160, [1, 7])
            net_b = slim.conv2d(net_b, 192, [7, 1])

            net = tf.concat([net_a, net_b], axis=3)
            net = slim.conv2d(net, 1154, [1, 1], activation_fn=None)

            net = x + net
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    with tf.name_scope('reduction_b'):
        net_a = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME')

        net_b = slim.conv2d(net, 256, [1, 1])
        net_b = slim.conv2d(net_b, 384, [3, 3], stride=2)

        net_c = slim.conv2d(net, 256, [1, 1])
        net_c = slim.conv2d(net_c, 288, [3, 3], stride=2)

        net_d = slim.conv2d(net, 256, [1, 1])
        net_d = slim.conv2d(net_d, 288, [3, 3])
        net_d = slim.conv2d(net_d, 320, [3, 3], stride=2)

        net = tf.concat([net_a, net_b, net_c, net_d], axis=3)
        net = tf.nn.relu(net)

    for i in range(5):
        with tf.name_scope('inception_resnet_c_%d' % i):
            x = slim.conv2d(net, 2048, [1, 1])

            net_a = slim.conv2d(net, 192, [1, 1])
            net_b = slim.conv2d(net, 192, [1, 1])
            net_b = slim.conv2d(net_b, 224, [1, 3])
            net_b = slim.conv2d(net_b, 256, [3, 1])
            net = tf.concat([net_a, net_b, net_c], axis=3)

            net = slim.conv2d(net, 2048, [1, 1], activation_fn=None)

            net = x + net
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    with tf.name_scope('average_pooling'):
        net = global_avg_pool(net)
        net = slim.flatten(net)
        tf.nn.dropout(net, 0.8)

    logits = slim.fully_connected(net, num_classes, activation_fn=tf.nn.softmax)

    return logits
