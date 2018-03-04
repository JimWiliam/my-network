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

import os, zipfile, fnmatch, shutil
import numpy as np
from scipy.misc import imread, imresize

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils as utils
from tensorflow_datasets.utils import to_nchw, recast, dequantize_and_scale, nchw_to_nhwc

# NOTE: If you decide to do a different train/test split, modify the count line below
settings = {
	'url': [
		('https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true', 'omniglot_train.zip'),
		('https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true', 'omniglot_test.zip')],
	'name': 'omniglot',
	'compression': tf_data.Compression.ZLIB,
	'count': {tf_data.Subset.TRAIN: 1623 * 16, tf_data.Subset.TEST: 1623 * 4},
	'size': (32, 32, 1),
	'classes': None
}

def process(train_path, test_path, size=(32, 32), shuffle=True, dequantize=True):

	# Unzip images
	basepath = os.path.dirname(train_path)
	with zipfile.ZipFile(train_path, 'r') as z:
		z.extractall(basepath)
	with zipfile.ZipFile(test_path, 'r') as z:
		z.extractall(basepath)

	background_path =  os.path.join(basepath, 'images_background/')
	evaluation_path =  os.path.join(basepath, 'images_evaluation/')
	
	train_image_paths = []
	train_course_labels = []
	train_fine_labels = []
	test_image_paths = []
	test_course_labels = []
	test_fine_labels = []

	# Navigate through script/symbol folders, recording paths/labels
	def navigate(path, current_labels = [-1, -1]):
		course_label = current_labels[0]
		fine_label = current_labels[1]

		for root, dirnames, filenames in os.walk(path):
			# We are in a script/symbol folder
			if len(dirnames) == 0:
				fine_label += 1
				for filename in fnmatch.filter(filenames, '*.png'):
					# Make the last four images belong to test set
					if filename.endswith(('17.png', '18.png', '19.png', '20.png')):
						test_image_paths.append(os.path.join(root, filename))
						test_fine_labels.append(fine_label)
						test_course_labels.append(course_label)
					else:
						train_image_paths.append(os.path.join(root, filename))
						train_fine_labels.append(fine_label)
						train_course_labels.append(course_label)

			# We are in a script folder
			elif dirnames[0].startswith('character'):
				course_label += 1

		return [course_label, fine_label]
	
	navigate(evaluation_path, navigate(background_path))

	count_train = len(train_image_paths)
	count_test = len(test_image_paths)
	train_x = np.zeros((count_train, 1) + size, dtype=np.float32)
	test_x = np.zeros((count_test, 1) + size, dtype=np.float32)

	def _load_image(fn):
		image = imread(fn, True)
		image = imresize(image, size, interp='bicubic')
		#image = image.reshape((-1))

		# NOTE: Reverse greyscale color
		image = np.abs(image-255.)

		return image

	# Create training data array
	print('Processing omniglot images')
	for idx, image_path in enumerate(train_image_paths):
		image = _load_image(image_path)
		train_x[idx, 0, :, :] = image

	# Create test data array
	for idx, image_path in enumerate(test_image_paths):
		image = _load_image(image_path)
		test_x[idx, 0, :, :] = image

	# Labels
	train_y = np.asarray(train_fine_labels, dtype=np.int64)
	test_y = np.asarray(test_fine_labels, dtype=np.int64)

	train_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(train_x), settings['size'])), dequantize)
	test_x = dequantize_and_scale(nchw_to_nhwc(to_nchw(recast(test_x), settings['size'])), dequantize)
	
	# Shuffle data
	if shuffle:
		train_x, train_y = utils.shuffle(train_x, train_y)
		test_x, test_y = utils.shuffle(test_x, test_y)

	# Remove unzipped images
	shutil.rmtree(background_path)
	shutil.rmtree(evaluation_path)

	# TODO: Apply additional processing here!

	return {
		tf_data.Subset.TRAIN: (train_x, train_y),
		tf_data.Subset.TEST: (test_x, test_y)}