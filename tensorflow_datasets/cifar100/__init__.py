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

import os, tarfile, shutil
import numpy as np

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils as utils
from tensorflow_datasets.utils import to_nchw, recast, dequantize_and_scale, nchw_to_nhwc

settings = {
	'url': ('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', 'cifar-100.tar.gz'),
	'name': 'cifar100',
	'compression': tf_data.Compression.ZLIB,
	'count': {tf_data.Subset.TRAIN: 50000, tf_data.Subset.TEST: 10000},
	'size': (32, 32, 3),
	'classes': 100
}

def process(filepath, dequantize=True):
	# Extract the tar if necessary to give the five training pickles and one test pickle
	batch_folder = 'cifar-100-python/'
	basepath = os.path.join(os.path.dirname(filepath), batch_folder)
	if not os.path.isfile(os.path.join(basepath, 'test')) \
			or not os.path.isfile(os.path.join(basepath, 'train')):
		with tarfile.open(filepath) as tar:
			tar.extractall(os.path.dirname(filepath))

	# Read in training data
	trainfile = os.path.join(basepath, 'train')
	unpickled = tf_data._unpickle(trainfile)
	train_x = unpickled['data']
	train_y = np.asarray(unpickled['fine_labels']) # 'course_labels'

	# Read in test data
	testfile = os.path.join(basepath, 'test')
	unpickled = tf_data._unpickle(testfile)
	test_x = unpickled['data']
	test_y = np.asarray(unpickled['fine_labels'])

	# Remove directory of untarred data
	#print('To be removed:', basepath)
	shutil.rmtree(basepath)

	# DEBUG: Checking whether values are integer valued
	#print(train_x[0])
	#raise Exception()

	train_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(train_x), settings['size'])), dequantize)
	test_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(test_x), settings['size'])), dequantize)

	# TODO: Additional processing

	return {
		tf_data.Subset.TRAIN: (train_x, train_y),
		tf_data.Subset.TEST: (test_x, test_y)}