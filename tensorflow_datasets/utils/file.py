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

import tensorflow_datasets as tf_data
import tensorflow_datasets.utils as utils
from six.moves import urllib
import os

def exists(filepath):
	return os.path.isfile(filepath)

def download(url, filepath):
	# If url is a string, then download a single file
	if utils.isstring(url):
		#filepath = os.path.join(path, filename)
		if not exists(filepath):
			wget(filepath, url)

	# Otherwise, assume it's a list/tuple of urls
	else:
		for url, filepath in zip(url, filename):
			#filepath = os.path.join(path, filename)
			if not exists(filepath):
				wget(filepath, url)

# Download a single file
# TODO: Timing
def wget(filepath, url):
	print('Downloading data from {0}'.format(url))
	urllib.request.urlretrieve(url, filepath)
	print('Download finished!')

def urlfile(url):
 return os.path.split(urllib.parse.urlsplit(url).path)[-1]
