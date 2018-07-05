import os
import sys
import argparse
import glob
import traceback
import time

import vn
import tensorflow as tf
import numpy as np
from mridata import VnMriReconstructionData
import mriutils

parser = argparse.ArgumentParser(description='reconstruct a given image data using a model')
parser.add_argument('image_config', type=str, help='config file for reconstruct')
parser.add_argument('model_name', type=str, help='name of the model in the log dir')
parser.add_argument('--o', dest='output_name', type=str, default='result', help='output name')
parser.add_argument('--epoch', type=int, default=None, help='epoch to evaluate')
parser.add_argument('--training_config', type=str, default='./configs/training.yaml', help='training config file')

if __name__ == '__main__':
    # parse the input arguments
    args = parser.parse_args()
    # image and model
    data_config = tf.contrib.icg.utils.loadYaml(args.image_config, ['data_config'])
    model_name = args.model_name

    output_name = args.output_name
    epoch = args.epoch

    checkpoint_config = tf.contrib.icg.utils.loadYaml(args.training_config, ['checkpoint_config'])

    all_models = glob.glob(checkpoint_config['log_dir'] + '/*')
    all_models = sorted([d.split('/')[-1] for d in all_models if os.path.isdir(d)])

    if not model_name in all_models:
        print('model not found in "{}"'.format(checkpoint_config['log_dir']))
        sys.exit(-1)

    ckpt_dir = checkpoint_config['log_dir'] + '/' + model_name + '/checkpoints/'
    eval_output_dir = checkpoint_config['log_dir'] + '/' + model_name + '/test/'

    with tf.Session() as sess:
        try:
            # load from checkpoint if required
            epoch = vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=epoch)
        except Exception as e:
            print(traceback.print_exc())

        # extract operators and variables from the graph
        u_op = tf.get_collection('u_op')[0]
        u_var = tf.get_collection('u_var')
        c_var = tf.get_collection('c_var')
        m_var = tf.get_collection('m_var')
        f_var = tf.get_collection('f_var')
        g_var = tf.get_collection('g_var')

        # create data object
        data = VnMriReconstructionData(data_config,
                                       u_var=u_var,
                                       f_var=f_var,
                                       c_var=c_var,
                                       m_var=m_var,
                                       g_var=g_var,
                                       load_eval_data=False,
                                       load_target=False)

        # run the model
        print('start reconstruction')
        eval_start_time = time.time()
        feed_dict, norm = data.get_test_feed_dict(data_config['dataset'],
                                                               data_config['dataset']['patient'],
                                                               data_config['dataset']['slice'],
                                                               return_norm=True)

        # get the reconstruction, re-normalize and postprocesss it
        u_i = sess.run(u_op, feed_dict=feed_dict)[0]
        u_i = u_i * norm  # renormalize
        u_i = mriutils.postprocess(u_i, data_config['dataset']['name'])
        time_reco = time.time() - eval_start_time
        print('reconstruction of {} image took {:f}s'.format(u_i.shape, time_reco))

        print('saving reconstructed image to "{}"'.format(output_name))
        # save mat file
        patient_id = '%s-p%d-sl%d' % (data_config['dataset']['name'],
                                      data_config['dataset']['patient'],
                                      data_config['dataset']['slice'])
        mriutils.saveAsMat(u_i, '%s-vn-%s.mat' % (output_name, patient_id), 'result_vn',
                  mat_dict={'normalization': np.asarray(norm)})

        # enhance image and save as png
        u_i_enhanced = mriutils.contrastStretching(np.abs(u_i), 0.002)
        mriutils.imsave(u_i_enhanced,
                        '%s-vn-%s.png' % (output_name, patient_id))
