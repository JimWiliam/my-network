# from tensorflow.contrib import slim
# import tensorflow as tf
#
#
# def inference(inputs, num_classes, n):
#     net = slim.conv2d(inputs, 16, [3, 3], 1)
#
#     for i in range(n):
#         res = net
#         net_a = slim.separable_conv2d(net, 8, [1, 1])
#         net_b = slim.separable_conv2d(net, 8, [3, 3])
#         net = tf.concat([net_a, net_b], axis=3)
#         net = net + res
#         net = slim.batch_norm(net, activation_fn=tf.nn.relu)
#
#     for i in range(n):
#         res = net
#         stride = 1
#         if i == 0:
#             stride = 2
#         net_a = slim.separable_conv2d(net, 16, [1, 1], stride=stride)
#         net_b = slim.separable_conv2d(net, 16, [3, 3], stride=stride)
#         net = tf.concat([net_a, net_b], axis=3)
#         if i == 0:
#             net = net + slim.separable_conv2d(res, 32, [1, 1], stride=stride)
#         else:
#             net = net + res
#         net = slim.batch_norm(net, activation_fn=tf.nn.relu)
#
#     for i in range(n):
#         res = net
#         stride = 1
#         if i == 0:
#             stride = 2
#         net_a = slim.separable_conv2d(net, 32, [1, 1], stride=stride)
#         net_b = slim.separable_conv2d(net, 32, [3, 3], stride=stride)
#         net = tf.concat([net_a, net_b], axis=3)
#         if i == 0:
#             net = net + slim.separable_conv2d(res, 64, [1, 1], stride=stride)
#         else:
#             net = net + res
#         net = slim.batch_norm(net, activation_fn=tf.nn.relu)
#
#     net = tf.reduce_mean(net, [1, 2])
#     net = slim.flatten(net)
#     logits = slim.fully_connected(net, num_classes, activation_fn=tf.nn.softmax)
#
#     return logits
