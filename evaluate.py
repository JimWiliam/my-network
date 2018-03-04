import tensorflow as tf


def top_k_error(logits, labels, k):
    total = labels.shape()[-1]
    incorrect = tf.nn.in_top_k(logits, labels, k)
    return total - tf.reduce_sum(tf.cast(incorrect, tf.int32))


def top_k_accuracy(logits, labels, k):
    correct = tf.nn.in_top_k(logits, labels, k)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


