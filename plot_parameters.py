import os
import sys
import argparse
import glob
import matplotlib
matplotlib.use('agg')
import vn.visualization
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description='plot parameters of a model')
parser.add_argument('model_name', type=str, help='name of the model in the log dir')
parser.add_argument('--epoch', type=int, default=None, help='epoch to evaluate')
parser.add_argument('--training_config', type=str, default='./configs/training.yaml', help='training config file')

if __name__ == '__main__':
    # parse the input arguments
    args = parser.parse_args()
    # image and model
    model_name = args.model_name

    # load the model
    checkpoint_config = tf.contrib.icg.utils.loadYaml(args.training_config, ['checkpoint_config'])

    all_models = glob.glob(checkpoint_config['log_dir'] + '/*')
    all_models = sorted([d.split('/')[-1] for d in all_models if os.path.isdir(d)])

    if not model_name in all_models:
        print('model not found in "{}"'.format(checkpoint_config['log_dir']))
        sys.exit(-1)

    # check the checkpoint directory
    ckpt_dir = os.path.expanduser(checkpoint_config['log_dir']) + '/' + model_name + '/checkpoints/'

    # one of the configs contains the network config
    configs = glob.glob(os.path.expanduser(checkpoint_config['log_dir']) + '/' + model_name + '/config/*.yaml')
    network_config = None
    for config in configs:
        if 'network' in open(config).read():
            network_config = tf.contrib.icg.utils.loadYaml(config, ['network'])
            break

    if network_config == None:
        print('no network config found in "{}""{}"/config'.format(checkpoint_config['log_dir'], model_name))
        sys.exit(-1)

    eval_output_dir = os.path.expanduser(checkpoint_config['log_dir']) + '/' + model_name + '/params/'
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    with tf.Session() as sess:
        epoch = vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=args.epoch)

        for var in tf.trainable_variables():
            var_name = var.name.split(':0')[0]
            print(var_name)

            if var_name == 'w1':
                x, phi = vn.visualization.extractActivationFunctionParams(var, network_config)
                num_kernels = phi.shape[1]
                num_stages = phi.shape[0]
                for stage in range(num_stages):
                    for kidx in range(num_kernels):
                        vn.visualization.saveSingleFunction(x, phi[stage,kidx], eval_output_dir, 'epoch%d_s%d_n%d' % (epoch, stage+1, kidx+1))

            elif var_name == 'k1':
                kernels = var.eval()
                num_kernels = kernels.shape[4]
                num_stages = kernels.shape[0]
                for stage in range(num_stages):
                    for kidx in range(num_kernels):
                        kernel = kernels[stage,:,:,0,kidx]
                        file_id_real = '%s/kernel_real_epoch%d_s%d_n%d.png' % (eval_output_dir, epoch, stage+1, kidx+1)
                        vn.visualization.saveSingleKernel(np.real(kernel), file_id_real)
                        file_id_imag = '%s/kernel_imag_epoch%d_s%d_n%d.png' % (eval_output_dir, epoch, stage+1, kidx+1)
                        vn.visualization.saveSingleKernel(np.imag(kernel), file_id_imag)

            elif var_name == 'lambda':
                print('lambda=',var.eval())
