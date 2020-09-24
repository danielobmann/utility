import numpy as np
import tensorflow as tf


def sob_weight(shape, srange=[-360, 360], beta=0.5):
    ret = np.zeros(shape)
    for i in range(shape[0]):
        ret[i, ...] = (1 + np.linspace(srange[0], srange[1], shape[1])**2)**beta
    return ret


def sob_loss(y_true, y_pred, srange=[-360, 360], beta=0.5):
    y = tf.cast(y_true - y_pred, tf.complex64)
    y = tf.squeeze(y)
    y = tf.fft(y)
    y = tf.square(tf.abs(y))
    s = sob_weight(shape=tf.keras.backend.int_shape(y_true)[1:3], srange=srange, beta=beta)
    s = tf.constant(s, tf.float32)
    y = y*s
    return tf.reduce_mean(y, name='sobolev_loss')


def l2(y_true, y_pred):
    return tf.reduce_mean(tf.squared_difference(y_true, y_pred), name='l2_loss')
