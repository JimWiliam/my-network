import os
import tensorflow as tf

os.environ['TF_DATA'] = '/media/todrip/数据/data/common'

weight_decay = 0.0001
momentum = 0.9
keep_prob = 0.8

init_learning_rate = 0.1
num_classes = 10
# reduction_ratio = 4

batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 300
image_size = 32
image_channels = 3

base_dir = '/media/todrip/数据/experiment'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/media/todrip/数据/experiment/cifar10_1',
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_string('network', 'my_resnet_20_13', """network to train""")
tf.app.flags.DEFINE_string('log_dir', '/media/todrip/数据/experiment/cifar10_1/logs', "log directory")
