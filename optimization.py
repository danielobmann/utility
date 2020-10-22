import tensorflow as tf
import numpy as np


def optimizer(loss, tvars, clipping=1.):
    total = int(np.sum([np.prod(t.shape) for t in tvars]))
    lr = tf.placeholder(tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    if clipping:
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clip_norm=clipping)
    else:
        grads = tf.gradients(loss, tvars)
    active = sum([tf.count_nonzero(grad) for grad in grads if grad is not None])/total
    train_op = opt.apply_gradients(zip(grads, tvars))
    return train_op, lr, active

