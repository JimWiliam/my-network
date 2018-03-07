# from tensorflow.contrib import slim
# import tensorflow as tf
#
#
# def inference(inputs, num_classes, n):
#     net = slim.conv2d(inputs, 16, [3, 3], 1)
#
#     for i in range(n):
#         res = net
#         net = slim.conv2d(net, 16, [3, 3])
#         net = slim.conv2d(net, 16, [3, 3])
#         net = net + res
#         net = slim.batch_norm(net, activation_fn=tf.nn.relu)
#
#     for i in range(n):
#         res = net
#         if i == 0:
#             net = slim.conv2d(net, 32, [3, 3], stride=2)
#         else:
#             net = slim.conv2d(net, 32, [3, 3])
#
#         net = slim.conv2d(net, 32, [3, 3])
#         if i == 0:
#             net = net + slim.conv2d(res, 32, [1, 1], stride=2)
#         else:
#             net = net + res
#         net = slim.batch_norm(net, activation_fn=tf.nn.relu)
#
#     for i in range(n):
#         res = net
#         if i == 0:
#             net = slim.conv2d(net, 64, [3, 3], stride=2)
#         else:
#             net = slim.conv2d(net, 64, [3, 3])
#
#         net = slim.conv2d(net, 64, [3, 3])
#         if i == 0:
#             net = net + slim.conv2d(res, 64, [1, 1], stride=2)
#         else:
#             net = net + res
#         net = slim.batch_norm(net, activation_fn=tf.nn.relu)
#
#
#     net = tf.reduce_mean(net, [1, 2])
#     net = slim.flatten(net)
#     logits = slim.fully_connected(net, num_classes, activation_fn=tf.nn.softmax)
#
#     return logits
