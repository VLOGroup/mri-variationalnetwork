import os
import shutil
import tensorflow as tf
import numpy as np

def setupLogDirs(suffix, args, checkpoint_config):
    ckpt_dir = checkpoint_config['log_dir'] + '/' + suffix + '/checkpoints'

    # create directories
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(checkpoint_config['log_dir'] + '/' + suffix + '/train'):
        os.makedirs(checkpoint_config['log_dir'] + '/' + suffix + '/train')
    if not os.path.exists(checkpoint_config['log_dir'] + '/' + suffix + '/config'):
        os.makedirs(checkpoint_config['log_dir'] + '/' + suffix + '/config')
    if not os.path.exists(checkpoint_config['log_dir'] + '/' + suffix + '/test'):
        os.makedirs(checkpoint_config['log_dir'] + '/' + suffix + '/test')

    # copy config files
    def copy(src, dst):
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copyfile(src, dst)

    copy(args.data_config, checkpoint_config['log_dir'] + '/' + suffix + '/config')
    copy(args.network_config, checkpoint_config['log_dir'] + '/' + suffix + '/config')
    copy(args.training_config, checkpoint_config['log_dir'] + '/' + suffix + '/config')
    try:
        copy(args.discriminator_config, checkpoint_config['log_dir'] + '/' + suffix + '/config')
    except:
        pass

def loadCheckpoint(sess, ckpt_dir, var_list=[], epoch=None, load_graph=True):
    ckpt_dir = os.path.expanduser(ckpt_dir)
    print('Checkpoint dir:', ckpt_dir)
    if not os.path.exists(ckpt_dir):
        raise RuntimeError("Checkpoint dir does not exist")
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)

    # load from checkpoint if required
    if epoch != None:
        model_checkpoint_path = ckpt_dir + 'checkpoint-%d' % epoch  # quick fix. loading checkpoint delievers the wrong model_checkpoint_path
    else: # Load last checkpoint
        model_checkpoint_path = checkpoint.model_checkpoint_path

    if load_graph:
        saver = tf.train.import_meta_graph('%s.meta' % model_checkpoint_path)
    else:
        saver = tf.train.Saver(var_list)

    if checkpoint and model_checkpoint_path:
        print("loading model from checkpoint", model_checkpoint_path)
        saver.restore(sess, model_checkpoint_path)
        print('finished')
        # get global number of steps
        start_epoch = int(model_checkpoint_path.split("-")[-1])
        print('start_epoch ', start_epoch)
        return start_epoch
    else:
        raise RuntimeError("Could not find checkpoint for epoch %s at %s" % (str(epoch),  ckpt_dir))

def ssim(input, target, ksize=11, sigma=1.5, L=1.0):
    def ssimKernel(ksize=ksize, sigma=sigma):
        if sigma == None:  # no gauss weighting
            kernel = np.ones((ksize, ksize, 1, 1)).astype(np.float32)
        else:
            x, y = np.mgrid[-ksize // 2 + 1:ksize // 2 + 1, -ksize // 2 + 1:ksize // 2 + 1]
            kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
            kernel = kernel[:, :, np.newaxis, np.newaxis].astype(np.float32)
        return kernel / np.sum(kernel)

    kernel = tf.Variable(ssimKernel(), name='ssim_kernel', trainable=False)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
    mu2 = tf.nn.conv2d(target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')
    mu1_sqr = mu1 ** 2
    mu2_sqr = mu2 ** 2
    mu1mu2 = mu1 * mu2
    sigma1_sqr = tf.nn.conv2d(input * input, kernel, strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC') - mu1_sqr
    sigma2_sqr = tf.nn.conv2d(target * target, kernel, strides=[1, 1, 1, 1], padding='VALID',
                              data_format='NHWC') - mu2_sqr
    sigma12 = tf.nn.conv2d(input * target, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC') - mu1mu2
    ssim_maps = ((2.0 * mu1mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sqr + mu2_sqr + C1) *
                                                                (sigma1_sqr + sigma2_sqr + C2))
    return tf.reduce_mean(tf.reduce_mean(ssim_maps, axis=(1, 2, 3)))
