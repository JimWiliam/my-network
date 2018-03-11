from hyper_parameter import *
import tensorflow_datasets as tf_data
from data_augmentation import *
from model import my_resnet

test_images, test_labels = tf_data.inputs_np(name='cifar100', subset=tf_data.Subset.TEST)
test_images = color_preprocessing(test_images)

test_labels = np.eye(np.max(test_labels, ) + 1)[test_labels]

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channels])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

logits = my_resnet.inference(x, num_classes=num_classes, n=5)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())
learning_rate = tf.placeholder(tf.float32)
training_flag = tf.placeholder(tf.bool)


def validate():
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('%s/model/%s.cpkt' % (FLAGS.train_dir, FLAGS.network))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return
        test_acc = 0.0
        test_pre_index = 0
        add = 1000
        for it in range(test_iteration):
            test_batch_x = test_images[test_pre_index: test_pre_index + add]
            test_batch_y = test_labels[test_pre_index: test_pre_index + add]
            test_pre_index = test_pre_index + add

            test_feed_dict = {
                x: test_batch_x,
                y: test_batch_y,
                learning_rate: None,
                training_flag: False
            }

            acc_ = sess.run([accuracy], feed_dict=test_feed_dict)

            test_acc += acc_
        test_acc /= test_iteration


if __name__ == '__main__':
    validate()
