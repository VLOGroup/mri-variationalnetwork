import time
import os
import vn

import tensorflow as tf
import argparse

from mridata import VnMriReconstructionData, VnMriFilenameProducer

import icg
import optotf

class VnMriReconstructionCell(icg.VnBasicCell):
    def mriForwardOpWithOS(self, u, coil_sens, sampling_mask):
        with tf.compat.v1.variable_scope('mriForwardOp'):
            # add frequency encoding oversampling
            pad_u = tf.cast(tf.multiply(tf.cast(tf.shape(sampling_mask)[1], tf.float32), 0.25) + 1, tf.int32)
            pad_l = tf.cast(tf.multiply(tf.cast(tf.shape(sampling_mask)[1], tf.float32), 0.25) - 1, tf.int32)
            u_pad = tf.pad(u, [[0, 0], [pad_u, pad_l], [0, 0]])
            u_pad = tf.expand_dims(u_pad, axis=1)
            # apply sensitivites
            coil_imgs = u_pad * coil_sens
            # centered Fourier transform
            Fu = icg.fftc2d(coil_imgs)
            # apply sampling mask
            mask = tf.expand_dims(sampling_mask, axis=1)
            kspace = tf.complex(tf.math.real(Fu) * mask, tf.math.imag(Fu) * mask)
        return kspace

    def mriAdjointOpWithOS(self, f, coil_sens, sampling_mask):
        with tf.compat.v1.variable_scope('mriAdjointOp'):
            # variables to remove frequency encoding oversampling
            pad_u = tf.cast(tf.multiply(tf.cast(tf.shape(sampling_mask)[1], tf.float32), 0.25) + 1, tf.int32)
            pad_l = tf.cast(tf.multiply(tf.cast(tf.shape(sampling_mask)[1], tf.float32), 0.25) - 1, tf.int32)
            # apply mask and perform inverse centered Fourier transform
            mask = tf.expand_dims(sampling_mask, axis=1)
            Finv = icg.ifftc2d(tf.complex(tf.math.real(f) * mask, tf.math.imag(f) * mask))
            # multiply coil images with sensitivities and sum up over channels
            img = tf.reduce_sum(Finv * tf.math.conj(coil_sens), 1)[:, pad_u:-pad_l, :]
        return img

    def mriForwardOp(self, u, coil_sens, sampling_mask):
        with tf.compat.v1.variable_scope('mriForwardOp'):
            # apply sensitivites
            coil_imgs = tf.expand_dims(u, axis=1) * coil_sens
            # centered Fourier transform
            Fu = icg.fftc2d(coil_imgs)
            # apply sampling mask
            mask = tf.expand_dims(sampling_mask, axis=1)
            kspace = tf.complex(tf.math.real(Fu) * mask, tf.math.imag(Fu) * mask)
        return kspace

    def mriAdjointOp(self, f, coil_sens, sampling_mask):
        with tf.compat.v1.variable_scope('mriAdjointOp'):
            # apply mask and perform inverse centered Fourier transform
            mask = tf.expand_dims(sampling_mask, axis=1)
            Finv = icg.ifftc2d(tf.complex(tf.math.real(f) * mask, tf.math.imag(f) * mask))
            # multiply coil images with sensitivities and sum up over channels
            img = tf.reduce_sum(Finv * tf.math.conj(coil_sens), 1)
        return img

    def call(self, t, inputs):
        # get the variables
        u_t_1 = inputs[0][t]

        # extract constants
        f = self._constants['f']
        c = self._constants['coil_sens']
        m = self._constants['sampling_mask']

        # get the parameters
        param_idx = self.time_to_param_index(t)

        # datatermweight
        lambdaa = self._params['lambda'][param_idx]
        # activation function weights
        w = self._params['w1'][param_idx]
        # convolution kernels
        k = self._params['k1'][param_idx]

        # extract options
        vmin = self._options['vmin']
        vmax = self._options['vmax']
        pad = self._options['pad']

        # split kernels
        k_real = tf.math.real(k)
        k_imag = tf.math.imag(k)

        # define the cell
        # pad the image to avoid problems at the border
        u_p = tf.pad(tf.expand_dims(u_t_1,-1), [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'SYMMETRIC')
        # split the image in real and imaginary part and perform convolution
        u_k_real = tf.nn.conv2d(tf.math.real(u_p), k_real, [1, 1, 1, 1], 'SAME')
        u_k_imag = tf.nn.conv2d(tf.math.imag(u_p), k_imag, [1, 1, 1, 1], 'SAME')
        # add up the convolution results
        u_k = u_k_real + u_k_imag
        # apply the activation functions
        shape = tf.shape(u_k)
        u_k = tf.transpose(tf.reshape(u_k, (-1, tf.shape(u_k)[-1])), [1, 0])
        u_k = tf.reshape(u_k, (tf.shape(u_k)[0], -1))
        f_u_k = optotf.activations._get_operator('rbf')(u_k, w, vmin=vmin, vmax=vmax)
        f_u_k = tf.reshape(tf.transpose(tf.reshape(f_u_k, tf.shape(u_k)), [1, 0]), shape)
        # perform transpose convolution for real and imaginary part
        u_k_T_real = tf.nn.conv2d_transpose(f_u_k, tf.math.real(k), tf.shape(u_p), [1, 1, 1, 1], 'SAME')
        u_k_T_imag = tf.nn.conv2d_transpose(f_u_k, tf.math.imag(k), tf.shape(u_p), [1, 1, 1, 1], 'SAME')
        # rebuild complex image
        u_k_T = tf.complex(u_k_T_real, u_k_T_imag)
        # remove padding
        Ru = u_k_T[:, pad:-pad, pad:-pad, 0]

        # normalize regularizer by number of filters
        Ru /= self._options['num_filter']

        # define dataterm operators according to sampling pattern
        if self._options['sampling_pattern'] == 'cartesian':
            print('mri op')
            forwardOp = self.mriForwardOp
            adjointOp = self.mriAdjointOp
        elif not 'sampling_pattern' in self._options or  self._options['sampling_pattern'] == 'cartesian_with_os':
            print('mri op with OS')
            forwardOp = self.mriForwardOpWithOS
            adjointOp = self.mriAdjointOpWithOS
        else:
            raise ValueError("Selected sampling pattern '%s' does not exist!" % (self._options['sampling_pattern']))

        # build dataterm
        Au = forwardOp(u_t_1, c, m)
        At_Au_f = adjointOp(Au - f, c, m)
        Du = tf.complex(tf.math.real(At_Au_f)*lambdaa, tf.math.imag(At_Au_f)*lambdaa)

        # gradient step
        u_t = u_t_1 - Ru - Du

        return [u_t]

if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str, default='./configs/training.yaml')
    parser.add_argument('--network_config', type=str, default='./configs/mri_vn.yaml')
    parser.add_argument('--data_config', type=str, default='./configs/data.yaml')
    parser.add_argument('--global_config', type=str, default='./configs/global.yaml')

    args = parser.parse_args()

    # Load the configs
    network_config, reg_config = icg.utils.loadYaml(args.network_config, ['network', 'reg'])
    checkpoint_config, optimizer_config = icg.utils.loadYaml(args.training_config, ['checkpoint_config', 'optimizer_config'])
    data_config = icg.utils.loadYaml(args.data_config, ['data_config'])
    global_config = icg.utils.loadYaml(args.global_config, ['global_config'])

    # Tensorflow config
    tf_config = tf.compat.v1.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = global_config['tf_allow_gpu_growth']

    # define the output locations
    base_name = os.path.basename(args.network_config).split('.')[0]
    suffix = base_name + '_' + time.strftime('%Y-%m-%d--%H-%M-%S')
    vn.setupLogDirs(suffix, args, checkpoint_config)

    # load data
    filename_producer = VnMriFilenameProducer(data_config)
    data = VnMriReconstructionData(data_config, filename_dequeue_op=filename_producer.dequeue_op, queue_capacity=global_config['data_queue_capacity'])

    network_config['sampling_pattern'] = data_config['sampling_pattern']

    # Create a queue runner that will run 4 threads in parallel to enqueue examples.
    qr_data = tf.train.QueueRunner(data.queue, [data.enqueue_op] * global_config['data_num_threads'])
    # Create a queue runner to produce the filenames
    qr_filenames = tf.train.QueueRunner(filename_producer.queue, [filename_producer.enqueue_op])

    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()

    # define parameters
    params = icg.utils.Params()
    const_params = icg.utils.ConstParams()

    vn.paramdefinitions.add_convolution_params(params, const_params, reg_config['filter1'])
    vn.paramdefinitions.add_activation_function_params(params, reg_config['activation1'])
    vn.paramdefinitions.add_dataterm_weights(params, network_config)

    # setup the network
    vn_cell = VnMriReconstructionCell(params=params.get(),
                              const_params=const_params.get(),
                              inputs=[data.u],
                              constants=data.constants,
                              options=network_config)

    mrirecon_vn = icg.VariationalNetwork(cell=vn_cell,
                                        num_stages=network_config['num_stages'],
                                        parallel_iterations=global_config['parallel_iterations'],
                                        swap_memory=global_config['swap_memory'])

    # get all images
    u_all = mrirecon_vn.get_outputs(stage_outputs=True)[0]
    u_T = tf.identity(u_all[-1], 'u_T')

    # define loss
    with tf.compat.v1.variable_scope('loss'):
        # mse abs-smoothed
        target_abs = tf.sqrt(tf.math.real((data.target) * tf.math.conj(data.target)) + 1e-12)
        output_abs = tf.sqrt(tf.math.real((u_T) * tf.math.conj(u_T)) + 1e-12)

        energy = tf.reduce_mean(tf.reduce_sum(((output_abs - target_abs) ** 2), axis=(1, 2)))

        # rmse
        denominator = tf.reduce_sum(tf.math.real((data.target) * tf.math.conj(data.target)), axis=(1, 2))
        nominator = tf.reduce_sum(tf.math.real((u_T - data.target) * tf.math.conj(u_T - data.target)), axis=(1, 2))
        rmse = tf.reduce_mean(tf.sqrt(nominator / denominator))

        # ssim
        output_abs = tf.expand_dims(tf.abs(u_T), -1)
        target_abs = tf.expand_dims(tf.abs(data.target), -1)
        L = tf.reduce_max(target_abs, axis=(1, 2, 3), keepdims=True) - tf.reduce_min(target_abs, axis=(1, 2, 3),
                                                                                     keepdims=True)
        ssim = vn.utils.ssim(output_abs, target_abs, L=L)

    # add images and energy to summary
    with tf.compat.v1.variable_scope('loss_summary'):
        tf.compat.v1.summary.scalar('energy', energy)
        tf.compat.v1.summary.scalar('rmse', rmse)
        tf.compat.v1.summary.scalar('ssim', ssim)

    # add images to tensorboard
    tf.compat.v1.summary.image('input', tf.abs(tf.expand_dims(data.u, -1)), max_outputs=10)
    for i in range(network_config['num_stages']):
        tf.compat.v1.summary.image('u%d' % (i + 1), tf.abs(tf.expand_dims(u_all[i + 1], -1)), max_outputs=10)
    tf.compat.v1.summary.image('target', tf.abs(tf.expand_dims(data.target, -1)), max_outputs=10)

    # define the optimizer
    optimizer = icg.optimizer.IPALMOptimizer(params, energy, optimizer_config)

    with tf.compat.v1.Session(config=tf_config) as sess:
        # initialize the variables
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # memorize a few ops and placeholders to be used in evaluation
        energy_op = tf.compat.v1.add_to_collection('energy_op', energy)
        ssim_op = tf.compat.v1.add_to_collection('ssim_op', ssim)
        rmse_op = tf.compat.v1.add_to_collection('rmse_op', rmse)
        u_op = tf.compat.v1.add_to_collection('u_op', u_all[-1])
        u_all_op = tf.compat.v1.add_to_collection('u_all_op', u_all)
        u_var = tf.compat.v1.add_to_collection('u_var', data.u)
        g_var = tf.compat.v1.add_to_collection('g_var', data.target)
        c_var = tf.compat.v1.add_to_collection('c_var', data.constants['coil_sens'])
        m_var = tf.compat.v1.add_to_collection('m_var', data.constants['sampling_mask'])
        f_var = tf.compat.v1.add_to_collection('f_var', data.constants['f'])
        g_var = tf.compat.v1.add_to_collection('g_var', data.target)

        # load from checkpoint if required
        saver = tf.compat.v1.train.Saver(max_to_keep=0)

        # initialize enqueuing threads
        enqueue_threads_filename_producer = qr_filenames.create_threads(sess, coord=coord, start=True)
        enqueue_threads_data = qr_data.create_threads(sess, coord=coord, start=True)

        # collect the summaries
        epoch_summaries = tf.compat.v1.summary.merge_all()
        image_summaries = tf.compat.v1.summary.merge_all(key='images')
        train_writer = tf.compat.v1.summary.FileWriter(checkpoint_config['log_dir'] + '/' + suffix + '/train/', sess.graph)

        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()

        iter_per_epoch = filename_producer.iter_per_epoch

        try:
            start_time = time.time()
            for epoch in range(0, optimizer_config['max_iter'] + 1):
                if coord.should_stop():
                    break

                # get next mini batch
                feed_dict = data.get_feed_dict(sess=sess)

                # run a single iteration
                optimizer.minimize(sess, epoch, feed_dict)

                feed_dict = data.get_eval_feed_dict()

                if (epoch % checkpoint_config['summary_modulo'] == 0) or epoch == optimizer_config['max_iter']:
                    summary = sess.run(epoch_summaries,
                                       feed_dict=feed_dict,
                                       options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % epoch)
                    train_writer.add_summary(summary, epoch)

                if (epoch % checkpoint_config['save_modulo'] == 0) or epoch == optimizer_config['max_iter']:
                    # update summary
                    summary = sess.run(image_summaries,
                                       feed_dict=feed_dict,
                                       options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'images%d' % epoch)
                    train_writer.add_summary(summary, epoch)
                    # save variables to checkpoint
                    saver.save(sess, checkpoint_config['log_dir'] + '/' + suffix + '/checkpoints/' + 'checkpoint', global_step=epoch)

                # compute the current energy
                e_i = sess.run(energy, feed_dict=feed_dict)
                print("epoch:", epoch, "energy =", e_i)

            print('Elapsed training time:', time.time() - start_time)

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        except KeyboardInterrupt as e:
            print('[KEYBOARD INTERRUPT]: Stop training.')
        finally:
            # Terminate as usual. It is innocuous to request stop twice.
            coord.request_stop()
            coord.join(enqueue_threads_data)
            coord.join(enqueue_threads_filename_producer)
