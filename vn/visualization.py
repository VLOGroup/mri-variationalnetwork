from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib

def resizeKernel(kernel, scaling):
    kernel_shape = list(kernel.shape)
    for idx in range(len(kernel_shape)):
        kernel_shape[idx] *= scaling
    kernel_shape = tuple(kernel_shape)

    return resize(kernel, kernel_shape, order=0, preserve_range=True)


def rescaleKernel(kernel):
    """ rescaleKernel to range [0,1]."""
    vmin = np.min(kernel)
    vmax = np.max(kernel)
    vabs = np.maximum(np.abs(vmin), np.abs(vmax))
    rescaled_kernel = (2.0 * vabs) / (vmax - vmin) * (kernel - vmin) - vabs
    rescaled_kernel = (1.0) / (2*vabs) * (rescaled_kernel + vabs)
    return rescaled_kernel

def saveSingleKernel(kernel, filename):
    kernel = rescaleKernel(kernel)
    kernel = resizeKernel(kernel, 4.0)
    plt.imsave(filename, kernel, cmap='gray', format='png')

def extractActivationFunctionParams(w, config):
    x_plt = np.linspace(config['vmin'], config['vmax'], 51, dtype=np.float32)
    x_plt = x_plt[:, np.newaxis]
    x_plt = np.tile(x_plt, (1, config['num_stages'] * config['num_filter']))
    x_plt_tf = tf.constant(x_plt, name='x_plt')

    phi_plt_f = tf.contrib.icg.activation_rbf(x_plt_tf,
                                              tf.reshape(w, (
                                              config['num_stages'] * config['num_filter'], config['num_weights'])),
                                              v_min=config['vmin'], v_max=config['vmax'],
                                              num_weights=config['num_weights'],
                                              feature_stride=1)
    phi_plt = tf.reshape(tf.transpose(phi_plt_f, (1, 0)),
                         (config['num_stages'], config['num_filter'], 51))
    x_plt = tf.reshape(tf.transpose(x_plt, (1, 0)),
                       (config['num_stages'], config['num_filter'], 51))
    return x_plt[0, 0].eval(), phi_plt.eval()


def plotSingleFunction(x, y, linewidth=12):
    opacity = 0.2
    matplotlib.rcParams.update({'font.size': 22})

    fig1, ax = plt.subplots()

    # Plot influence function
    ax.plot(x, y, linewidth=linewidth)
    ax.set_frame_on(False)
    #    ax.set_xlabel(r'$y$')
    #    ax.set_ylabel(r"$\phi^\prime(y)$")
    ax.set_xlim([min(x), -min(x)])
    ax.axhline(y=0, color='k', alpha=opacity, linewidth=linewidth)
    ax.axvline(x=0, color='k', alpha=opacity, linewidth=linewidth)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig2, ax = plt.subplots()

    # Plot influence function
    rho = np.cumsum(y)
    rho -= np.min(rho)
    # print rho.max()
    rho /= len(x)
    rho *= (x[1] - x[0])
    ax.plot(x, rho, 'r', linewidth=linewidth)
    ax.set_frame_on(False)
    #    ax.set_xlabel(r'$y$')
    #    ax.set_ylabel(r'$\phi(y)$')
    ax.set_xlim([min(x), -min(x)])
    ax.axhline(y=0, color='k', alpha=opacity, linewidth=linewidth)
    ax.axvline(x=0, color='k', alpha=opacity, linewidth=linewidth)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig1, fig2


def saveSingleFunction(x, y, output_dir, file_id):
    fig1, fig2 = plotSingleFunction(x, y)
    filename1 = '%s/activation_%s.png' % (output_dir, file_id)
    filename2 = '%s/penalty_%s.png' % (output_dir, file_id)
    fig1.savefig(filename1, bbox_inches='tight')
    fig2.savefig(filename2, bbox_inches='tight')
