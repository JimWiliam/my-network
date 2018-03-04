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

import numpy as np

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils as utils
import tensorflow_datasets.mnist as mnist

settings = {
	'url': 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz',
	'name': 'mnist.binarized',
	'compression': tf_data.Compression.ZLIB,
	'count': {tf_data.Subset.TRAIN: 60000, tf_data.Subset.TEST: 10000},
	'size': (28, 28, 1),
	'classes': 10
}

def process(filepath):
	subsets = mnist.process(filepath)
	train_x, train_y = subsets[tf_data.Subset.TRAIN]
	test_x, test_y = subsets[tf_data.Subset.TEST]

	train_x = tf_data._binarize(train_x)
	test_x = tf_data._binarize(test_x)

	# TODO: Apply additional processing here

	return {
		tf_data.Subset.TRAIN: (train_x, train_y),
		tf_data.Subset.TEST: (test_x, test_y)}