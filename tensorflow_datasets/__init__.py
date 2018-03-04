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

from enum import Enum
import numpy as np

import gzip, os, sys, six
import importlib
from six.moves import cPickle as pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# import tensorflow_datasets.utils as utils
from tensorflow_datasets.utils import isstring
from tensorflow_datasets.utils.file import exists, download, urlfile
from tensorflow_datasets.utils.list import wrap_list


class Compression(Enum):
    NONE = 1
    GZIP = 2
    ZLIB = 3


class Subset(Enum):
    TRAIN = 1
    TEST = 2
    VALIDATION = 3


compression_suffix = {
    Compression.NONE: '',
    Compression.GZIP: 'gzip',
    Compression.ZLIB: 'zlib'
}

subset_suffix = {
    Subset.TRAIN: 'train',
    Subset.TEST: 'test',
    Subset.VALIDATION: 'validation'
}

# These modules names are forbidden for being used as data sets
reserved_names = set(['utils'])


# Search folders under data to enumerate all datasets
def enumerate():
    # Work out what the path to the data folder is
    path = os.path.dirname(os.path.abspath(__file__))
    len_path = len(path.split(os.sep))

    # Walk over it recording which folders are valid modules
    datasets = []
    for root, dirnames, filenames in os.walk(path):
        if root != path:
            if any([f == '__init__.py' for f in filenames]):
                name = os.path.join(*root.split(os.sep)[len_path:]).replace('/', '.').replace('\\', '.')
                if not name in reserved_names:
                    datasets.append(name)

    return datasets


def download_all():
    datasets = enumerate()
    for name in datasets:
        for subset in [Subset.TRAIN, Subset.TEST]:
            check_exists(name=name, subset=subset)


# Get the number of the training/test samples, as reported by the settings
def count(name='mnist', subset=Subset.TRAIN):
    settings = _settings(name)
    return settings['count'][subset]


# Import the dataset module and retrieve settings
def _settings(name):
    assert not name in reserved_names

    dataset = importlib.import_module('tensorflow_datasets.' + name)
    return dataset.settings


# Return the HWC shape of a sample image after transformations have been applied!
def unflattened_sample_shape(settings):
    # print('settings', settings)
    shape = list(_settings(settings['dataset'])['size'])
    if 'transformations' in settings and settings['transformations']:
        for k, v in six.viewitems(settings['transformations']):
            if k == 'resize':
                shape[0:2] = v
    # TODO: Extra case once added cropping
    # elif k == 'crop':

    return shape


def sample_shape(settings):
    shape = unflattened_sample_shape(settings)

    if settings['transformations'] and 'flatten' in settings['transformations']:
        shape = [np.prod(shape)]

    return shape


# Retrieve the process method from a dataset module
def _process(name):
    assert not name in reserved_names

    dataset = importlib.import_module('tensorflow_datasets.' + name)
    return dataset.process


def _unpickle(filepath):
    with open(filepath, "rb") as f:
        try:
            return pickle.load(f, encoding='latin1')
        except:
            return pickle.load(f)


def _compression_options(compression):
    if compression == Compression.NONE:
        return None
    elif compression == Compression.GZIP:
        return tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    elif compression == Compression.ZLIB:
        return tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    else:
        raise ValueError('Invalid compression type passed to "compression_options()"')


def _read_and_decode_image(
        name,
        filename_queue,
        return_labels=True,
        enqueue_many_size=1000,
        batch_size=100,
        num_threads=2,
        min_after_dequeue=10000,
        transformations=None):
    settings = _settings(name)
    reader = tf.TFRecordReader(options=_compression_options(settings['compression']))
    # queue_batch = []
    # for i in range(enqueue_many_size):
    #	_, serialized_example = reader.read(filename_queue)
    #	queue_batch.append(serialized_example)
    _, queue_batch = reader.read_up_to(filename_queue, enqueue_many_size)

    batch_serialized_example = tf.train.shuffle_batch(
        [queue_batch],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=num_threads * batch_size * 10 + min_after_dequeue,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True)

    if return_labels:
        features = tf.parse_example(
            batch_serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64)
            })
    else:
        features = tf.parse_example(
            batch_serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([], tf.string)
            })

    # Convert from a scalar string tensor to a tensor
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image.set_shape([batch_size, np.prod(settings['size'])])

    # print('image size:', [batch_size, np.prod(settings['size'])])
    # print('argument to tf.reshape:', [batch_size] + list(settings['size']))

    # TODO: Applying more transformations and in separate function!
    # TODO: Transformation that converts from (channel, width, height) into (width, height, channel)
    image = transformations_op(transformations, image, settings, batch_size=batch_size)

    # TODO: Converting labels to one-hot encoding on the fly

    image = tf.identity(image, name='samples')
    if return_labels:
        # TODO: One-hot encodings if specified
        label = tf.cast(features['label'], tf.int32, name='labels')
        return image, label
    else:
        return image


def transformations_op(transformations, image, settings, batch_size=None):
    if transformations:
        # NOTE: Even if flattened, transformations should still work!
        # if not 'flatten' in transformations:
        image = tf.reshape(image, [batch_size] + list(settings['size']))

        for k, v in six.viewitems(transformations):
            if k == 'rescale':
                a, b = v
                # print('Applying normalization', v)
                assert (a < b)
                image = image * (b - a) + a
            elif k == 'resize':
                image = tf.image.resize_images(image, v)
            elif not (k == 'flatten' or k == 'binarize'):
                raise ValueError('Invalid transformation value given to transformations_op()')

        # NOTE: Need to do binarization after resizing
        if 'binarize' in transformations:
            # NOTE: They keep changing the argument names...
            try:
                dist = tf.contrib.distributions.Bernoulli(p=image)
            except:
                dist = tf.contrib.distributions.Bernoulli(probs=image)
            image = tf.cast(dist.sample(), tf.float32)

        if 'flatten' in transformations:
            image = tf.reshape(image, [batch_size, -1])
    else:
        # print('Resizing to {}'.format([batch_size] + list(settings['size'])))
        image = tf.reshape(image, [batch_size] + list(settings['size']))
    return image


def _filepath(name, subset, path, tfrecords=True):
    settings = _settings(name)
    if tfrecords:
        compression_part = '.' + compression_suffix[settings['compression']] if settings[
                                                                                    'compression'] != Compression.NONE else ''
        filepath = os.path.join(path, settings['name'] + '.' + subset_suffix[subset] + compression_part + '.tfrecords')
    else:
        filepath = os.path.join(path, settings['name'] + '.' + subset_suffix[subset] + '.pkl.gz')
    return filepath


def _download_raw(name, path):
    settings = _settings(name)

    # Works whether single url or list
    # NOTE: wrap_list wraps tuples as well, which is what we want here
    for url in wrap_list(settings['url']):
        # Case where url filename specifies disk filename
        if isstring(url):
            raw_filename = urlfile(url)

        # Case where filename is different from url filename
        else:
            url, raw_filename = url

        raw_filepath = os.path.join(path, raw_filename)

        # TODO: Exception handling!
        # Download raw data if necessary
        if not exists(raw_filepath):
            download(url, raw_filepath)


def _url_filepaths(url, path):
    urls = wrap_list(url)
    filepaths = []

    for url in urls:
        if isstring(url):
            filepaths.append(os.path.join(path, urlfile(url)))
        else:
            filepaths.append(os.path.join(path, url[1]))

    return filepaths


# Check that .tfrecords files exist and produce any missing ones
def check_exists(name='mnist', subset=Subset.TRAIN, path=os.environ['TF_DATA'], tfrecords=True):
    settings = _settings(name)
    process = _process(name)

    if not exists(_filepath(name, subset, path, tfrecords)):
        # Make sure we have all the raw files to process
        _download_raw(name, path)

        # Return numpy arrays for the train and test subsets
        data = process(*_url_filepaths(settings['url'], path))

        # NOTE: Only create .tfrecords/.pkl.gz that don't already exist, but test both subsets/data formats
        for subset in data.keys():
            for tfrecords in [True, False]:
                filepath = _filepath(name, subset, path, tfrecords)

                if not exists(filepath):
                    subset_data = data[subset]
                    if tfrecords:
                        _convert(subset_data[0], subset_data[1], filepath, settings['compression'])
                    else:
                        print('Writing', filepath)
                        _pickle_zip(subset_data, filepath)


def _pickle_zip(subset_data, filepath):
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(subset_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _unzip_unpickle(filepath):
    with gzip.open(filepath) as f:
        try:
            return pickle.load(f, encoding='latin1')
        except:
            return pickle.load(f)


# Load data as a NumPy array
def inputs_np(name='mnist', subset=Subset.TRAIN, path=os.environ['TF_DATA'], return_labels=True, transformations=None):
    settings = _settings(name)

    check_exists(name=name, subset=subset, path=path, tfrecords=False)
    filepath = _filepath(name, subset, path, tfrecords=False)

    print('Loading dataset {}'.format(filepath))
    images, labels = _unzip_unpickle(filepath)
    num_examples = images.shape[0]

    # Apply transformations
    if transformations and 'flatten' in transformations:
        shape = (num_examples, np.prod(settings['size']))
        images = images.reshape(shape)
    else:
        shape = (num_examples,) + settings['size']

    if transformations:
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=shape)
        image_op = transformations_op(transformations, image_placeholder, settings, batch_size=num_examples)
        with tf.InteractiveSession().as_default():
            images = image_op.eval(feed_dict={image_placeholder: images})

    if return_labels:
        return images, labels
    else:
        return images


# Create input nodes in computational graph
# NOTE: You should pin these operations to the CPU
# TODO: Make args variable and pass to input_node
def inputs(name='mnist', subset=Subset.TRAIN, path=os.environ['TF_DATA'], return_labels=True, batch_size=100,
           num_threads=4, transformations=None):
    # Check that .tfrecords files exist and download if necessary
    check_exists(name=name, subset=subset, path=path)

    # Create nodes in graph
    return _input_node(
        name=name,
        subset=subset,
        path=path,
        return_labels=return_labels,
        batch_size=batch_size,
        num_threads=num_threads,
        transformations=transformations
    )


def _input_node(name, subset, path, return_labels, batch_size, num_threads, transformations):
    # Note that an tf.train.QueueRunner is added to the graph, which must be run using e.g. tf.train.start_queue_runners()
    tfrecords_filepath = _filepath(name, subset, path)

    # Check that file exists
    if not os.path.exists(tfrecords_filepath):
        raise IOError('Data file does not exist: {}'.format(tfrecords_filepath))

    print('Loading dataset {}'.format(tfrecords_filepath))

    # NOTE: After experimenting with num_epochs I found the only way to avoid annoying warning messages is to limit the number of epochs yourself!
    filename_queue = tf.train.string_input_producer([tfrecords_filepath], num_epochs=None)

    # Even when reading in multiple threads, share the filename queue.
    return _read_and_decode_image(
        name,
        filename_queue,
        return_labels=return_labels,
        # enqueue_many_size=10,
        batch_size=batch_size,
        # num_threads=num_threads,
        # min_after_dequeue=1000,
        transformations=transformations)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert(images, labels, filepath, compression):
    """Converts a dataset to tfrecords."""
    num_examples = images.shape[0]
    # rows = images.shape[1]
    # cols = images.shape[2]
    # depth = images.shape[3]

    # print('num_examples', num_examples)
    # print('images.shape:', images.shape)

    # filepath = os.path.join(paths['data'], name + '.tfrecords')
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath, options=_compression_options(compression))
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


# Converts data on unit interval by sampling from bernoulli with parameters given by data
def _binarize(x):
    return np.random.binomial(1, x, size=x.shape).astype(np.float32)
