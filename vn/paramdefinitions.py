import numpy as np
import tensorflow as tf
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tensorflow.contrib import icg

from . import proxmaps


def plt_act_function(x, phi):
    my_dpi = 96.
    fig = matplotlib.figure.Figure(figsize=(350/my_dpi, 250/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(1, 1, 1)
    images = []
    for s in range(phi.shape[0]):
        images_stage = []
        for i in range(phi.shape[1]):
            ax.clear()
            ax.plot(x, phi[s, i, :])
            if fig.canvas is None:
                FigureCanvasAgg(fig)
            fig.canvas.draw()
            img_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images_stage.append(img_data)
        images.append(images_stage)
    return np.asarray(images)


def add_activation_function_params(params, config):

    print('activation function {}'.format(config['name']))

    # activation function
    x_0 = np.linspace(config['vmin'], config['vmax'], config['num_weights'], dtype=np.float32)
    print('  %s init %f' % (config['init_type'],config['init_scale']))
    if config['init_type'] == 'linear':
        w_0 = config['init_scale']*x_0
    elif config['init_type'] == 'tv':
        w_0 = config['init_scale'] * np.sign(x_0)
    elif config['init_type'] == 'relu':
        w_0 = config['init_scale'] * np.maximum(x_0, 0)
    elif config['init_type'] == 'student-t':
        alpha = 100
        w_0 = config['init_scale'] * np.sqrt(alpha)*x_0/(1+0.5*alpha*x_0**2)
    else:
        raise ValueError("init_type '%s' not defined!" % config['init_type'])
    w_0 = w_0[np.newaxis,:]
    w_stage = np.tile(w_0, (config['num_stages'], config['num_filter'], 1))
    w = tf.Variable(initial_value=w_stage, dtype=tf.float32, name=config['name'])

    prox_w = None

    params.add(w, prox=prox_w)

    # add kernels to summary
    with tf.variable_scope('activation_plot'):
        x_plt = np.linspace(config['vmin'], config['vmax'], 151, dtype=np.float32)
        x_plt = x_plt[:, np.newaxis]
        x_plt = np.tile(x_plt, (1, config['num_stages']*config['num_filter']))
        x_plt_tf = tf.constant(x_plt, name='x_plt')

        w_r = tf.reshape(w, (config['num_stages']*config['num_filter'], config['num_weights']))

        phi_plt_f = icg.activation_rbf(x_plt_tf, w_r,
                                       v_min=config['vmin'], v_max=config['vmax'], num_weights=config['num_weights'],
                                       feature_stride=1)
        phi_plt = tf.reshape(tf.transpose(phi_plt_f, (1, 0)),
                             (config['num_stages'], config['num_filter'], 151))
        phi_img = tf.py_func(plt_act_function, [x_plt_tf[:, 0], phi_plt], tf.uint8)

    for i in range(config['num_stages']):
        tf.summary.image('phi_' + config['name'] + '_%d' % (i + 1), phi_img[i], max_outputs=config['num_filter'], collections=['images'])


def add_convolution_params(params, const_params, config):
    def generate_random_numbers(config, zero_mean=True):
        init = np.random.randn(config['num_stages'],
                               config['filter_size'],
                               config['filter_size'],
                               config['features_in'],
                               config['features_out']).astype(np.float32) / \
               np.sqrt(config['filter_size'] ** 2 * config['features_in'])
        if zero_mean:
            init -= np.mean(init, axis=(1, 2, 3), keepdims=True)

        return init

    # define prox calculations
    if 'prox_zero_mean' in config and config['prox_zero_mean'] == False:
        prox_zero_mean = False
    else:
        prox_zero_mean = True

    if 'prox_norm' in config and config['prox_norm'] == False:
        prox_norm = False
    else:
        prox_norm = True

    print('kernel {}'.format(config['name']))
    print('  prox_zero_mean: ', prox_zero_mean)
    print('  prox_norm: ', prox_norm)

    # filter kernels
    k_0 = generate_random_numbers(config) + 1j * generate_random_numbers(config)
    k = tf.Variable(initial_value=k_0, dtype=tf.complex64, name=config['name'])

    prox_k = proxmaps.zero_mean_norm_ball(k, zero_mean=prox_zero_mean, normalize=prox_norm, axis=(1,2,3))

    params.add(k, prox=prox_k)

    # add kernels to summary
    def get_kernel_img(k):
        _, _, n_f_in, n_f_out = k.shape
        k_img = tf.concat([tf.concat([k[:, :, in_f, out_f] for in_f in range(n_f_in)], axis=0)
                           for out_f in range(n_f_out)], axis=1)
        k_img = tf.expand_dims(tf.expand_dims(k_img, -1), 0)
        return k_img

    with tf.variable_scope('kernel_%s_summary' % config['name']):
        for i in range(config['num_stages']):
            tf.summary.image('%s_%d_real' % (config['name'], i + 1), get_kernel_img(tf.real(k[i])), collections=['images'])
            tf.summary.image('%s_%d_imag' % (config['name'], i + 1), get_kernel_img(tf.imag(k[i])), collections=['images'])

def add_dataterm_weights(params, config):
    lambda_init = 0.1
    if 'datatermweight_init' in config.keys():
        lambda_init = config['datatermweight_init']
    # dataterm weight
    lambda_0 = np.ones((config['num_stages'], 1), dtype=np.float32) * lambda_init
    lambdaa = tf.Variable(initial_value=lambda_0, dtype=tf.float32, name='lambda')
    # define constraints
    # positivity of lambda
    with tf.variable_scope('prox_lambda'):
        prox_lambda = tf.assign(lambdaa, tf.maximum(lambdaa, 0))

    params.add(lambdaa, prox=prox_lambda)

    # add lambda plots to tf summary
    for i in range(config['num_stages']):
        tf.summary.scalar('lambda%d' % (i + 1), lambdaa[i, 0], collections=['images'])
