import os
import numpy as np

import vn

import tensorflow as tf
import argparse
import glob
import traceback

from mridata import VnMriReconstructionData
import mriutils
import icg

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str, default='./configs/training.yaml')
    parser.add_argument('--data_config', type=str, default='./configs/data.yaml')
    parser.add_argument('--epoch', type=int, default=None) # takes the last available epoch

    args = parser.parse_args()
    checkpoint_config = icg.utils.loadYaml(args.training_config, ['checkpoint_config'])
    data_config = icg.utils.loadYaml(args.data_config, ['data_config'])

    eval_datasets = data_config['dataset']

    all_folders = glob.glob(checkpoint_config['log_dir'] + '/*')
    all_folders = sorted([d for d in all_folders if os.path.isdir(d)])

    save_output = True
    disp_slice_eval = False

    for suffix in all_folders:

        tf.reset_default_graph()
        suffix = suffix.split('/')[-1]
        print(suffix)
        # check the checkpoint directory
        ckpt_dir = checkpoint_config['log_dir'] + '/' + suffix + '/checkpoints/'
        eval_output_dir = checkpoint_config['log_dir'] + '/' + suffix + '/test/'

        with tf.compat.v1.Session() as sess:
            try:
                # load from checkpoint if required
                epoch = vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=args.epoch)
            except Exception as e:
                print(traceback.print_exc())
                continue

            # extract a few ops and variables to be used in evaluation
            u_op = tf.compat.v1.get_collection('u_op')[0]
            u_var = tf.compat.v1.get_collection('u_var')
            g_var = tf.compat.v1.get_collection('g_var')
            c_var = tf.compat.v1.get_collection('c_var')
            m_var = tf.compat.v1.get_collection('m_var')
            f_var = tf.compat.v1.get_collection('f_var')

            # create data object
            data = VnMriReconstructionData(data_config,
                                           u_var=u_var,
                                           f_var=f_var,
                                           g_var=g_var,
                                           c_var=c_var,
                                           m_var=m_var,
                                           load_eval_data=False)

            # Evaluate the performance
            for dataset in eval_datasets:
                eval_patients = dataset['eval_patients']
                if not os.path.exists(eval_output_dir + '/%s' % dataset['name']):
                    os.makedirs(eval_output_dir + '/%s' % dataset['name'])

                print("Evaluating performance {:s} for {:s}, epoch {:d}".format(suffix, dataset['name'], epoch))

                ssim_eval_dataset = []
                rmse_eval_dataset = []

                for patient in eval_patients:
                    path = os.path.expanduser(data_config['base_dir'] + '/' + dataset['name'] + '/')

                    if not os.path.exists(path + '/%d' % patient):
                        print('  Eval path %s , patient %d does not exist. Continue...' % (path, patient))
                        continue
                    else:
                        print('  Eval path %s , patient %d' % (path, patient))

                    num_slices = len(glob.glob(path + '/%d/rawdata*.mat' % patient))

                    output = []
                    target = []
                    input0 = []
                    normalization = []

                    # build volume
                    for idx in range(1, num_slices+1):
                        feed_dict, norm = data.get_test_feed_dict(dataset, patient, idx, return_norm=True)

                        u_i = sess.run(u_op, feed_dict=feed_dict)

                        # re-normalize images
                        output.append(u_i[0] * norm)
                        target.append(feed_dict[data.target][0] * norm)
                        input0.append(feed_dict[data.u][0] * norm)
                        normalization.append(norm)

                    # postprocess images
                    output = mriutils.postprocess(np.asarray(output), dataset['name'])
                    target = mriutils.postprocess(np.asarray(target), dataset['name'])
                    input0 = mriutils.postprocess(np.asarray(input0), dataset['name'])

                    # evaluation
                    ssim_patient = mriutils.ssim(output, target)
                    rmse_patient = mriutils.rmse(output, target)
                    ssim_eval_dataset.append(ssim_patient)
                    rmse_eval_dataset.append(rmse_patient)

                    print("    Patient {:d}: {:8.4f} {:8.4f}".format(patient, rmse_patient, ssim_patient))

                    output_path = '%s/%s/%d/' % (eval_output_dir, dataset['name'], patient)
                    mriutils.saveAsMat(output,  '%s/vn-%d.mat' % (output_path, epoch), 'result_vn',
                              mat_dict={'normalization': np.asarray(normalization)})
                    mriutils.saveAsMat(target, (output_path, epoch), '%s/reference.mat', 'reference',
                              mat_dict={'normalization': np.asarray(normalization)})
                    mriutils.saveAsMat(input0, (output_path, epoch), '%s/zero_filling.mat', 'result_zf',
                              mat_dict={'normalization': np.asarray(normalization)})

                print("  Dataset {:s}: {:8.4f} {:8.4f}".format(dataset['name'],
                                  np.mean(rmse_eval_dataset),
                                  np.mean(ssim_eval_dataset)
                                  ))
