# MIT License
#
# Copyright (c) 2017, Stefan Webb. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hyper_parameter
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils.plot as plot


# A minimal example of how to read data using the tensorflow-datasets library
# Reads in a single minibatch from the Omniglot dataset and outputs a 

def run(name='mnist'):
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        tf.set_random_seed(0)
        np.random.seed(0)

        # Input images and labels.
        with tf.device("/cpu:0"):
            # images, labels = tf_data.inputs(name=name, return_labels=True, transformations=None) # transformations={'flatten': None}
            # images, labels = tf_data.inputs(name=name, return_labels=True, transformations={'rescale': (-1., 1.), 'resize': (32, 32)})
            images, labels = tf_data.inputs(name=name, return_labels=True, transformations=None)

        # with tf.device("/gpu:0"):

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        with tf.Session() as sess:
            # Initialize the variables (the trained variables and the epoch counter)
            sess.run(init_op)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # NOTE: You need all this code to use queue runners, otherwise your program will hang
            with coord.stop_on_exception():
                while not coord.should_stop():
                    # Stuff to do before shutting down
                    images_batch, labels_batch = sess.run([images, labels])
                    coord.request_stop()

            coord.join(threads)

        # Examine the batch we've extracted
    print('size of image minibatch', images.shape)

    # plot.sample_grid('{}_samples'.format(name), images_batch, imgrange=(-1, 1.))
    # plot.sample_grid('{}_samples'.format(name), images)
    # plot.save_labels('{}_labels'.format(name), labels)


if __name__ == '__main__':
    datasets = tf_data.enumerate()
    # datasets = ['celeba']

    for name in datasets:
        run(name)
