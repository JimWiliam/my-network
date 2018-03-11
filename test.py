# from hyper_parameter import *
# import tensorflow_datasets as tf_data
# from model.xception import xception
# from data_augmentation import *
# from model import my_resnet
# from model import s_resnet
#
# images, labels = tf_data.inputs_np(name="cifar10", subset=tf_data.Subset.TRAIN)
# test_images, test_labels = tf_data.inputs_np(name='cifar10', subset=tf_data.Subset.TEST)
# images = color_preprocessing(images)
# test_images = color_preprocessing(test_images)
#
# # labels = np.eye(np.max(labels) + 1)[labels]
# # test_labels = np.eye(np.max(test_labels, ) + 1)[test_labels]
#
# x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channels])
# y = tf.placeholder(tf.int32, shape=[None])
# y_ = tf.one_hot(y, depth=num_classes)
#
# learning_rate = tf.placeholder(tf.float32)
# training_flag = tf.placeholder(tf.bool)
#
# logits = my_resnet.inference(x, num_classes=num_classes, n=3)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
#
# l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
# train_op = optimizer.minimize(cost + l2_loss * weight_decay)
#
# saver = tf.train.Saver(tf.global_variables())
#
# cr = tf.argmax(logits, 1, output_type=tf.int32)
# correct_prediction = tf.equal(cr, y)
# cast = tf.cast(correct_prediction, tf.float32)
# accuracy = tf.reduce_mean(cast)
#
# with tf.Session() as sess:
#     ckpt = tf.train.get_checkpoint_state('%s/model/%s.cpkt' % (FLAGS.train_dir, FLAGS.network))
#     if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
#         saver.restore(sess, ckpt.model_checkpoint_path)
#     else:
#         sess.run(tf.global_variables_initializer())
#
#     summary_writer = tf.summary.FileWriter('%s/%s' % (FLAGS.log_dir, FLAGS.network))
#
#     epoch_learning_rate = init_learning_rate
#
#     batch_x = images[0: 64]
#     batch_y = labels[0: 64]
#
#     train_feed_dict = {
#         x: batch_x,
#         y: batch_y,
#         learning_rate: epoch_learning_rate,
#         training_flag: True
#     }
#
#     cr1, y1, y_1, acc1, correct_prediction1, cast1 = sess.run(
#         [cr, y, y_, accuracy, correct_prediction, cast],
#         feed_dict=train_feed_dict)
#
#     print(cr1)
#     print(y1)
#     print(y_1)
#     print(acc1)
#     print(correct_prediction1)
#     print(cast1)


print(5//2)