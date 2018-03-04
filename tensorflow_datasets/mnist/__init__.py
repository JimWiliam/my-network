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
from tensorflow_datasets.utils import to_nchw, recast, dequantize_and_scale, nchw_to_nhwc

settings = {
	'url': 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz',
	'name': 'mnist',
	'compression': tf_data.Compression.ZLIB,
	'count': {tf_data.Subset.TRAIN: 60000, tf_data.Subset.TEST: 10000},
	'size': (28, 28, 1),
	'classes': 10,
}

def process(filepath, subtract_mean=False, dequantize=False):
	# Unzip dataset and 
	train_set, valid_set, test_set = tf_data._unzip_unpickle(filepath)

	test_x, test_y = test_set
	valid_x, valid_y = valid_set
	train_x, train_y = train_set

	train_x = np.concatenate([train_x, valid_x])
	train_y = np.concatenate([train_y, valid_y])

	#print(dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(train_x), settings['size'])), False).dtype)

	train_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(train_x*256.), settings['size'])), dequantize)
	test_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(test_x*256.), settings['size'])), dequantize)

	#print(train_x.dtype)
	#print(test_x.dtype)

	#print(type(train_y))
	#print(type(test_y))

	# DEBUG: Checking whether dequantization has taken place
	#print(np.min(train_x[0]*256.), np.max(train_x[0]*256.))
	#raise Exception()

	# TODO: Apply additional processing here!

	return {
		tf_data.Subset.TRAIN: (train_x, train_y),
		tf_data.Subset.TEST: (test_x, test_y)}
