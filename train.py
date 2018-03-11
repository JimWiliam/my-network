import time
from hyper_parameter import *
import tensorflow_datasets as tf_data
from model.xception import xception
from data_augmentation import *
from model import my_resnet
from model import s_resnet
from model import my_dense_net

images, labels = tf_data.inputs_np(name="cifar10", subset=tf_data.Subset.TRAIN)
test_images, test_labels = tf_data.inputs_np(name='cifar10', subset=tf_data.Subset.TEST)
images = color_preprocessing(images)
test_images = color_preprocessing(test_images)

# labels = np.eye(np.max(labels) + 1)[labels]
# test_labels = np.eye(np.max(test_labels, ) + 1)[test_labels]

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channels])
y = tf.placeholder(tf.int32, shape=[None])
y_ = tf.one_hot(y, depth=num_classes)

learning_rate = tf.placeholder(tf.float32)
training_flag = tf.placeholder(tf.bool)

logits = my_resnet.inference(x, num_classes=10, n=3)
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits)
total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum,
                                       use_nesterov=True)
train_op = optimizer.minimize(total_loss)

saver = tf.train.Saver(tf.global_variables())

correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
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

        loss_, acc_ = sess.run([total_loss, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


def train():
    # config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('%s/model/%s/%s.cpkt' % (FLAGS.train_dir, FLAGS.network, FLAGS.network))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch = 0
        else:
            sess.run(tf.global_variables_initializer())
            epoch = 0
        summary_writer = tf.summary.FileWriter('%s/%s' % (FLAGS.log_dir, FLAGS.network))

        while epoch < total_epochs:
            epoch += 1
            if epoch < 180:
                epoch_learning_rate = 0.1
            elif epoch < 250:
                epoch_learning_rate = 0.01
            else:
                epoch_learning_rate = 0.001

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):
                if pre_index + batch_size < 50000:
                    batch_x = images[pre_index: pre_index + batch_size]
                    batch_y = labels[pre_index: pre_index + batch_size]
                else:
                    batch_x = images[pre_index:]
                    batch_y = labels[pre_index:]
                batch_x = data_augmentation(batch_x)
                train_feed_dict = {
                    x: batch_x,
                    y: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss, batch_acc = sess.run([train_op, total_loss, accuracy], feed_dict=train_feed_dict)
                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

            train_loss /= iteration  # average loss
            train_acc /= iteration  # average accuracy

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = evaluate(sess)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            line = "%s: epoch %d/%d, train_loss %.4f, train_acc %.4f, test_loss %.4f, test_acc %.4f" % (
                t, epoch, total_epochs, train_loss, train_acc,
                test_loss, test_acc)
            print(line)

            output = open('%s/%s/log.txt' % (FLAGS.log_dir, FLAGS.network), 'a')
            output.write(line)
            output.write('\n')
            output.close()
        saver.save(sess=sess, save_path='%s/model/%s.ckpt' % (FLAGS.train_dir, FLAGS.network))


if __name__ == "__main__":
    train()
