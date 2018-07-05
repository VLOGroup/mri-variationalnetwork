import tensorflow as tf
import numpy as np

def zero_mean_norm_ball(x, zero_mean=True, normalize=True, norm_bound=1.0, norm='l2', mask=None, axis=(0, ...)):
    """ project onto zero-mean and norm-one ball
    :param x: tf variable which should be projected
    :param zero_mean: boolean True for zero-mean. default=True
    :param normalize: boolean True for l_2-norm ball projection. default:True
    :param norm_bound: defines the size of the norm ball
    :param norm: type of the norm
    :param mask: binary mask to compute the mean and norm
    :param axis: defines the axis for the reduction (mean and norm)
    :return: projection ops
    """

    if mask is None:
        shape = []
        for i in range(len(x.shape)):
            if i in axis:
                shape.append(x.shape[i])
            else:
                shape.append(1)
        mask = tf.ones(shape, dtype=np.float32)

    with tf.variable_scope('prox_' + x.name.split(':')[0]):
        x_masked = tf.complex(tf.real(x) * mask, tf.imag(x) * mask)

        if zero_mean:
            x_mean_real = tf.reduce_sum(tf.real(x_masked), axis=axis, keepdims=True) / tf.reduce_sum(mask, axis=axis,
                                                                                                     keepdims=True)
            x_mean_imag = tf.reduce_sum(tf.imag(x_masked), axis=axis, keepdims=True) / tf.reduce_sum(mask, axis=axis,
                                                                                                     keepdims=True)
            x_mean = tf.complex(x_mean_real * mask, x_mean_imag * mask)
            x_zm = x_masked - x_mean
        else:
            x_zm = x_masked

        if normalize:
            if norm == 'l2':
                x_proj = tf.assign(x, x_zm / tf.complex(tf.maximum(tf.sqrt(tf.reduce_sum(tf.real(x_zm * tf.conj(x_zm)),
                                                                                         axis=axis, keepdims=True)) /
                                                                   norm_bound, 1), tf.zeros_like(x_zm, tf.float32)))
            else:
                raise ValueError("Norm '%s' not defined." % norm)
        elif zero_mean:
            x_proj = tf.assign(x, x_zm)
        else:
            x_proj = None

    return x_proj
