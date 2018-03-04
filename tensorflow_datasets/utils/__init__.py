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

import six
import numpy as np

# TODO: Better way to do this?
#import tensorflow_datasets.utils.file as file
#import tensorflow_datasets.utils.list as list
#import tensorflow_datasets.utils.plot as plot
#import tensorflow_datasets.utils.timer as timer

def isstring(s):
	return isinstance(s, six.string_types)

# Converts data format of a numpy vector from NCHW to NHWC
def nchw_to_nhwc(batch):
	return np.transpose(batch, axes=(0, 2, 3, 1))

# Given shape in HWC
def nchw_shape(size):
	return np.concatenate(([-1], np.flip(size, 0)))

# TODO: Move into a separate module for numpy operations?
def flip_axes(tensor):
	axes = np.arange(len(tensor.shape))
	return np.moveaxis(tensor, axes, np.flip(axes, 0))

def flip_image(tensor):
	assert len(tensor.shape) == 4
	return tensor.swapaxes(2, 3)

# TODO: Some way to specify whether using 32- or 64-bit floating point arithmetic
def recast(x):
	return np.asarray(x, dtype=np.float32)

def to_nchw(x, size):
	return x.reshape(nchw_shape(size))

def dequantize_and_scale(x, dequantize):
	if dequantize:
		return (x + np.random.random_sample(x.shape).astype(np.float32)) / 256.
	else:
		return x / 255.

# Shuffle multiple numpy arrays by their first index
def shuffle(*args):
	idx = np.random.permutation(np.arange(args[0].shape[0]))
	new_args = []
	for arg in args:
		new_args.append(arg[idx])

	return new_args