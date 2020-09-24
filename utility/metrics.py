import tensorflow as tf
import numpy as np


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(x_result, x_true, name='psnr'):
    with tf.name_scope(name):
        maxval = tf.reduce_max(x_true) - tf.reduce_min(x_true)
        mse = tf.reduce_mean((x_result - x_true) ** 2)
        return 20 * log10(maxval) - 10 * log10(mse)


def psnr_numpy(x_result, x_true):
    maxval = np.amax(x_true) - np.amin(x_true)
    mse = np.mean((x_true - x_result)**2)
    return 20*np.log10(maxval) - 10*np.log10(mse)


def nmse(x_result, x_true, name='nmse'):
    with tf.name_scope(name):
        error = tf.reduce_sum((x_result - x_true) ** 2, axis=[1, 2, 3])
        normalizer = tf.reduce_sum(x_true**2, axis=[1, 2, 3])
        return tf.reduce_mean(error/normalizer)


def nmse_numpy(x_result, x_true):
    error = np.mean((x_true - x_result)**2)
    normalizer = np.mean(x_true**2)
    return error/normalizer
