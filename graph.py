from hyper_parameter import *
from model import my_resnet
from model import resnet
from model import my_dense_net

tf.app.flags.DEFINE_string('graph', 'my_dense_net', 'model graph')

x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, image_channels], name='imputs')

logits = my_dense_net.inference(x, num_classes=num_classes, n=9, k=12)

# logits = my_resnet.inference(x, num_classes=num_classes, n=10)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('/media/todrip/数据/experiment/models/%s/%s' % (FLAGS.graph, FLAGS.graph))
    summary_writer.add_graph(sess.graph)
    num_parameters = 0
    for var in tf.trainable_variables():
        n = 1
        for i in var.shape.as_list():
            n *= i
        num_parameters += n
    print(num_parameters)
