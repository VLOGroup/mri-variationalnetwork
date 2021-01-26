import tensorflow as tf

def conv2d_complex(u, k, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d convolution with the same interface as `conv2d`.
    """
    conv_rr = tf.nn.conv2d(tf.math.real(u), tf.math.real(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ii = tf.nn.conv2d(tf.math.imag(u), tf.math.imag(k),  strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ri = tf.nn.conv2d(tf.math.real(u), tf.math.imag(k), strides=strides, padding=padding,
                                     data_format=data_format)
    conv_ir = tf.nn.conv2d(tf.math.imag(u), tf.math.real(k), strides=strides, padding=padding,
                                     data_format=data_format)
    return tf.complex(conv_rr-conv_ii, conv_ri+conv_ir)

def conv2d_transpose_complex(u, k, output_shape, strides=[1,1,1,1], padding='SAME', data_format='NHWC'):
    """ Complex 2d transposed convolution with the same interface as `conv2d_transpose`.
    """
    convT_rr = tf.nn.conv2d_transpose(tf.math.real(u), tf.math.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ii = tf.nn.conv2d_transpose(tf.math.imag(u), tf.math.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ri = tf.nn.conv2d_transpose(tf.math.real(u), tf.math.imag(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    convT_ir = tf.nn.conv2d_transpose(tf.math.imag(u), tf.math.real(k), output_shape, strides=strides, padding=padding,
                                     data_format=data_format)
    return tf.complex(convT_rr+convT_ii, convT_ir-convT_ri)

def ifftc2d(inp):
    """ Centered inverse 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = tf.sqrt(tf.cast(numel, tf.float32))

    out = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(inp, axes=(-2,-1))), axes=(-2,-1))
    out = tf.complex(tf.math.real(out)*scale, tf.math.imag(out)*scale)
    return out

def fftc2d(inp):
    """ Centered 2d Fourier transform, performed on axis (-1,-2).
    """
    shape = tf.shape(inp)
    numel = shape[-2]*shape[-1]
    scale = 1.0 / tf.sqrt(tf.cast(numel, tf.float32))

    out = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(inp, axes=(-2,-1))), axes=(-2,-1))
    out = tf.complex(tf.math.real(out) * scale, tf.math.imag(out) * scale)
    return out