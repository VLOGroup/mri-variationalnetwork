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
import icg

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
    data_config = icg.utils.loadYaml(args.image_config, ['data_config'])
    model_name = args.model_name

    output_name = args.output_name
    epoch = args.epoch

    checkpoint_config = icg.utils.loadYaml(args.training_config, ['checkpoint_config'])

    all_models = glob.glob(checkpoint_config['log_dir'] + '/*')
    all_models = sorted([d.split('/')[-1] for d in all_models if os.path.isdir(d)])

    if not model_name in all_models:
        print('model not found in "{}"'.format(checkpoint_config['log_dir']))
        sys.exit(-1)

    ckpt_dir = checkpoint_config['log_dir'] + '/' + model_name + '/checkpoints/'
    eval_output_dir = checkpoint_config['log_dir'] + '/' + model_name + '/test/'

    with tf.compat.v1.Session() as sess:
        try:
            # load from checkpoint if required
            epoch = vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=epoch)
        except Exception as e:
            print(traceback.print_exc())

        # extract operators and variables from the graph
        u_op = tf.compat.v1.get_collection('u_op')[0]
        u_var = tf.compat.v1.get_collection('u_var')
        c_var = tf.compat.v1.get_collection('c_var')
        m_var = tf.compat.v1.get_collection('m_var')
        f_var = tf.compat.v1.get_collection('f_var')
        g_var = tf.compat.v1.get_collection('g_var')

        # create data object
        data = VnMriReconstructionData(data_config,
                                       u_var=u_var,
                                       f_var=f_var,
                                       c_var=c_var,
                                       m_var=m_var,
                                       g_var=g_var,
                                       load_eval_data=False,
                                       load_target=True)

        # run the model
        print('start reconstruction')
        eval_start_time = time.time()
        feed_dict, norm = data.get_test_feed_dict(data_config['dataset'],
                                                  data_config['dataset']['patient'],
                                                  data_config['dataset']['slice'],
                                                  return_norm=True)

        # get the reconstruction, re-normalize and postprocesss it
        u_i = sess.run(u_op, feed_dict=feed_dict)[0]
        u_i = u_i * norm
        u_i = mriutils.postprocess(u_i, data_config['dataset']['name'])

        # target
        target = feed_dict[data.target][0]*norm
        target = mriutils.postprocess(target, data_config['dataset']['name'])

        # zero filling
        zero_filling = feed_dict[data.u][0]*norm
        zero_filling = mriutils.postprocess(zero_filling, data_config['dataset']['name'])

        # evaluation
        rmse_vn = mriutils.rmse(u_i, target)
        rmse_zf = mriutils.rmse(zero_filling, target)
        ssim_vn = mriutils.ssim(u_i, target)
        ssim_zf = mriutils.ssim(zero_filling, target)

        print("Zero filling: RMSE={:.4f} SSIM={:.4f}  VN: RMSE={:.4f} SSIM={:.4f}".format(rmse_zf, ssim_zf, rmse_vn, ssim_vn))

        time_reco = time.time() - eval_start_time
        print('reconstruction of {} image took {:f}s'.format(u_i.shape, time_reco))

        print('saving reconstructed image to "{}"'.format(output_name))
        # save mat files
        patient_id = '%s-p%d-sl%d' % (data_config['dataset']['name'],
                                      data_config['dataset']['patient'],
                                      data_config['dataset']['slice'])
        mriutils.saveAsMat(u_i, '%s-vn-%s' % (output_name, patient_id), 'result_vn',
                  mat_dict={'normalization': np.asarray(norm)})
        mriutils.saveAsMat(zero_filling, '%s-zf-%s' % (output_name, patient_id), 'result_zf',
                  mat_dict={'normalization': np.asarray(norm)})
        mriutils.saveAsMat(target, '%s-ref-%s' % (output_name, patient_id), 'reference',
                  mat_dict={'normalization': np.asarray(norm)})

        # enhance image with same parameters for all images
        v_min, v_max = mriutils.getContrastStretchingLimits(np.abs(target),
                                                            saturated_pixel=0.002)
        target_enhanced = mriutils.normalize(np.abs(target), v_min=v_min, v_max=v_max)
        u_i_enhanced = mriutils.normalize(np.abs(u_i), v_min=v_min, v_max=v_max)
        zf_enhanced = mriutils.normalize(np.abs(zero_filling), v_min=v_min, v_max=v_max)

        # save pngs
        mriutils.imsave(u_i_enhanced,
                        '%s-vn-%s.png' % (output_name, patient_id))

        mriutils.imsave(target_enhanced,
                        '%s-ref-%s.png' % (output_name, patient_id))

        mriutils.imsave(zf_enhanced,
                        '%s-zf-%s.png' % (output_name, patient_id))
