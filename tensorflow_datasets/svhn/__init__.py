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

from scipy.io import loadmat
import numpy as np

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils as utils
from tensorflow_datasets.utils import to_nchw, recast, dequantize_and_scale, nchw_to_nhwc, flip_axes, flip_image

# TODO: Saving files under a different name, e.g. svhn_train.mat, svhn_test.mat
settings = {
	'url': [
		('http://ufldl.stanford.edu/housenumbers/train_32x32.mat', 'svhn_train.mat'),
		('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', 'svhn_test.mat')],
	'name': 'svhn',
	'compression': tf_data.Compression.ZLIB,
	'count': {tf_data.Subset.TRAIN: 73257, tf_data.Subset.TEST: 26032},
	'size': (32, 32, 3),
	'classes': 10
}

def process(train_filepath, test_filepath, dequantize=True, shuffle=True):
	train = loadmat(train_filepath)
	train_x = train['X']	
	train_y = train['y'].reshape((-1))
	train_y[train_y == 10] = 0

	test = loadmat(test_filepath)
	test_x = test['X']
	test_y = test['y'].reshape((-1))
	test_y[test_y == 10] = 0

	train_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(flip_image(flip_axes(train_x))), settings['size'])), dequantize)
	test_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(flip_image(flip_axes(test_x))), settings['size'])), dequantize)

	if shuffle:
		train_x, train_y = utils.shuffle(train_x, train_y)
		test_x, test_y = utils.shuffle(test_x, test_y)

	# TODO: Apply additional processing here!

	return {
		tf_data.Subset.TRAIN: (train_x, train_y),
		tf_data.Subset.TEST: (test_x, test_y)}