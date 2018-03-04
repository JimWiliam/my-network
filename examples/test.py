import hyper_parameter
import tensorflow_datasets as tf_data
import tensorflow as tf

train_x, train_y = tf_data.inputs(name='cifar100')
test_x, test_y = tf_data.inputs(name='cifar100', subset=tf_data.Subset.TEST, batch_size=10000)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



# with tf.Session() as sess:
#     # Initialize the variables (the trained variables and the epoch counter)
#     sess.run(init_op)
#     # Start input enqueue threads.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     # NOTE: You need all this code to use queue runners, otherwise your program will hang
#     with coord.stop_on_exception():
#         while not coord.should_stop():
#             # Stuff to do before shutting down
#             train_images, train_labels = sess.run([train_x, train_y])
#             test_images, test_labels = sess.run([test_x, test_y])
#             # coord.request_stop()
#     coord.join(threads)


