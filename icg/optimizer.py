import tensorflow as tf
import numpy as np

from tensorflow.python.client import timeline

class IPALMOptimizer(object):
    """ Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems
        T. Pock and S. Sabach
        SIAM J. Imaging Sci., 9(4), 1756–1787, 2016.
        Read More: http://epubs.siam.org/doi/abs/10.1137/16M1064064?journalCode=sjisbi

        Args:
            params: Parameters to optimize (class defined in utils)
            energy: Energy to minimize
            config: optimization config (max_bt_iter, fixed_momentum, reset_memory, L_init)
    """

    def __init__(self, params, energy, config, name='IPALM'):
        self._params = params
        self._energy = energy

        self._gradients = []
        self._grad_step = []
        self._grad_var = []
        self._var_old = []
        self._L_np = []
        self._update_var = []
        self._epoch = 0
        self._var_old_np = []
        self._max_bt_iter = config['max_bt_iter']
        self._momentum = config['fixed_momentum']
        self._reset_memory = config['reset_memory']

        with tf.compat.v1.variable_scope(name):
            for param_idx in range(len(params.get(as_dict=False))):
                current_param = params.get(as_dict=False)[param_idx]

                self._grad_var.append(tf.compat.v1.placeholder(current_param.dtype, shape=current_param.get_shape(),
                                               name='grad_var_' + current_param.name.split(':')[0]))
                self._var_old.append(tf.compat.v1.placeholder(current_param.dtype, shape=current_param.get_shape(),
                                              name='var_old_' + current_param.name.split(':')[0]))

                if 'L_init' in config:
                    self._L_np.append(config['L_init'])
                else:
                    self._L_np.append(1000000.0)

                self._gradients.append(tf.gradients(energy, current_param)[0])
                self._update_var.append(tf.compat.v1.assign(current_param, self._var_old[param_idx]))

    def minimize(self, sess, epoch, feed_dict):
        self._epoch = epoch

        if epoch == 0:
            print('Reset var_old')
            self._var_old_np = []
            for param_idx in range(len(self._params.get(as_dict=False))):
                if self._params.get_prox()[param_idx] is not None:
                    if self._params.get_tau()[param_idx] is not None:
                          sess.run(self._params.get_prox()[param_idx],
                                 feed_dict={self._params.get_tau()[param_idx]: 1./self._L_np[param_idx]})
                    else:
                        sess.run(self._params.get_prox()[param_idx])
                var_old_np = self._params.get(as_dict=False)[param_idx].eval(session=sess)
                if self._params.get_np_prox()[param_idx] is not None:
                    var_old_np = self._params.get_np_prox()[param_idx](var_old_np, 1./self._L_np[param_idx])
                self._var_old_np.append(var_old_np)

        if self._momentum == None:
            beta = (np.mod(epoch, self._reset_memory)) / (np.mod(epoch, self._reset_memory) + 3.0)
            print("beta:", beta)
        else:
            beta = self._momentum

        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
#
        for param_idx in range(len(self._params.get(as_dict=False))):
            # overrelaxation and store old theta value
            theta_np = self._params.get(as_dict=False)[param_idx].eval(session=sess)
            theta_tilde_np = theta_np + beta * (theta_np - self._var_old_np[param_idx])
            self._var_old_np[param_idx] = theta_np.copy()

            # update param and compute energy, gradient
            sess.run(self._update_var[param_idx], feed_dict={self._var_old[param_idx]: theta_tilde_np})
            if epoch == 0:
                energy_old, gradient_val = sess.run([self._energy, self._gradients[param_idx]],
                                                    feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('timeline_%s_%d.json' % (self._params.get(as_dict=False)[param_idx].name, epoch), 'w') as f:
                    f.write(ctf)
            else:
                energy_old, gradient_val = sess.run([self._energy, self._gradients[param_idx]], feed_dict=feed_dict)

            # backtracking
            for bt_iter in range(1, self._max_bt_iter + 1):
                # do the gradient step with the current step size
                theta_np = theta_tilde_np - gradient_val/self._L_np[param_idx]
                # do a np prox map
                if self._params.get_np_prox()[param_idx] is not None:
                    theta_np = self._params.get_np_prox()[param_idx](theta_np, 1./self._L_np[param_idx])

                # update the tf variables
                sess.run(self._update_var[param_idx], feed_dict={self._var_old[param_idx]: theta_np})
                # do a tf prox map
                if self._params.get_prox()[param_idx] is not None:
                    if self._params.get_tau()[param_idx] is not None:
                        sess.run(self._params.get_prox()[param_idx],
                                 feed_dict={self._params.get_tau()[param_idx]: 1./self._L_np[param_idx]})
                    else:
                        sess.run(self._params.get_prox()[param_idx])

                Q_lhs = sess.run(self._energy, feed_dict=feed_dict)

                theta_new_np = self._params.get(as_dict=False)[param_idx].eval(session=sess)

                Q_rhs = energy_old + np.real(np.sum((theta_new_np - theta_tilde_np) * np.conj(gradient_val))) +\
                                  self._L_np[param_idx] / 2.0 * np.linalg.norm(theta_new_np - theta_tilde_np) ** 2

                delta = 1 + np.sign(Q_rhs) * 1e-3
                if Q_lhs <= Q_rhs * delta:
                    self._L_np[param_idx] *= 0.75
                    if self._L_np[param_idx] <= 1e-3:
                        self._L_np[param_idx] = 1e-3
                    break
                else:
                    self._L_np[param_idx] *= 2.0
                    if self._L_np[param_idx] > 1e12:
                        self._L_np[param_idx] = 1e12
                        break

            print('*** L=%f, param=%s, Qlhs=%f, Qrhs=%f ||g||=%e' % (
            self._L_np[param_idx], self._params.get(as_dict=False)[param_idx].name, Q_lhs, Q_rhs, np.linalg.norm(gradient_val)))


class StageIPALMOptimizer(object):
    """ Inertial Proximal Alternating Linearized Minimization (iPALM) for Nonconvex and Nonsmooth Problems
        T. Pock and S. Sabach
        SIAM J. Imaging Sci., 9(4), 1756–1787, 2016.
        Read More: http://epubs.siam.org/doi/abs/10.1137/16M1064064?journalCode=sjisbi

        The StageIPALM optimizer can be used for the variationalnetwork.

        Args:
            num_stages: Number of stages of the variational network
            params: Parameters to optimize (class defined in utils)
            energy: Energy to minimize
            config: optimization config (max_bt_iter, fixed_momentum, reset_memory, L_init)
    """
    def __init__(self, num_stages, params, energy, config, name='StageIPALM'):
        self._params = params
        self._energy = energy

        self._num_stages = num_stages

        self._gradients = []
        self._grad_step = []
        self._grad_var = []
        self._var_old = []
        self._L_np = []
        self._update_var = []
        self._epoch = 0
        self._var_old_np = []
        self._max_bt_iter = config['max_bt_iter']
        self._momentum = config['fixed_momentum']
        self._reset_memory = config['reset_memory']

        with tf.compat.v1.variable_scope(name):
            for param_idx in range(len(params.get(as_dict=False))):
                current_param = params.get(as_dict=False)[param_idx]

                self._grad_var.append(tf.compat.v1.placeholder(current_param.dtype, shape=current_param.get_shape(),
                                                     name='grad_var_' + current_param.name.split(':')[0]))
                self._var_old.append(tf.compat.v1.placeholder(current_param.dtype, shape=current_param.get_shape(),
                                              name='var_old_' + current_param.name.split(':')[0]))

                if 'L_init' in config:
                    self._L_np.append([config['L_init']] * self._num_stages)
                else:
                    self._L_np.append([1000000.0] * self._num_stages)

                self._gradients.append(tf.gradients(energy, current_param)[0])
                self._update_var.append(tf.compat.v1.assign(current_param, self._var_old[param_idx]))

    def minimize(self, sess, epoch, feed_dict):
        self._epoch = epoch

        if epoch == 0:
            print('Reset var_old')
            self._var_old_np = []
            for param_idx in range(len(self._params.get(as_dict=False))):
                if self._params.get_prox()[param_idx] is not None:
                    if self._params.get_tau()[param_idx] is not None:
                          sess.run(self._params.get_prox()[param_idx],
                                 feed_dict={self._params.get_tau()[param_idx]: 1./self._L_np[param_idx][0]})
                    else:
                        sess.run(self._params.get_prox()[param_idx])
                var_old_np = self._params.get(as_dict=False)[param_idx].eval(session=sess)
                if self._params.get_np_prox()[param_idx] is not None:
                    var_old_np = self._params.get_np_prox()[param_idx](var_old_np, 1. / self._L_np[param_idx][0])
                self._var_old_np.append(var_old_np)

        if self._momentum == None:
            beta = (np.mod(epoch, self._reset_memory)) / (np.mod(epoch, self._reset_memory) + 3.0)
            print("beta:", beta)
        else:
            beta = self._momentum

        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
#
        for s in range(self._num_stages):
            for param_idx in range(len(self._params.get(as_dict=False))):
                # overrelaxation and store old theta value
                theta_np = self._params.get(as_dict=False)[param_idx].eval(session=sess)
                theta_tilde_np = theta_np.copy()
                theta_tilde_np[s] = theta_np[s] + beta * (theta_np[s] - self._var_old_np[param_idx][s])
                self._var_old_np[param_idx][s] = theta_np[s].copy()

                # update param and compute energy, gradient
                sess.run(self._update_var[param_idx], feed_dict={self._var_old[param_idx]: theta_tilde_np})
                energy_old, gradient_val = sess.run([self._energy, self._gradients[param_idx]], feed_dict=feed_dict)
                # clear the gradient for all non-active stages
                clear = np.zeros_like(gradient_val)
                clear[s, ...] = 1
                gradient_val *= clear

                # backtracking
                for bt_iter in range(1, self._max_bt_iter + 1):
                    # do the gradient step with the current step size
                    theta_np = theta_tilde_np - gradient_val / self._L_np[param_idx][s]
                    # do a np prox map
                    if self._params.get_np_prox()[param_idx] is not None:
                        theta_np = self._params.get_np_prox()[param_idx](theta_np, 1. / self._L_np[param_idx][s])

                    # update the tf variables
                    sess.run(self._update_var[param_idx], feed_dict={self._var_old[param_idx]: theta_np})
                    # do a tf prox map
                    if self._params.get_prox()[param_idx] is not None:
                        if self._params.get_tau()[param_idx] is not None:
                            sess.run(self._params.get_prox()[param_idx],
                                     feed_dict={self._params.get_tau()[param_idx]: 1./self._L_np[param_idx][s]})
                        else:
                            sess.run(self._params.get_prox()[param_idx])

                    Q_lhs = sess.run(self._energy, feed_dict=feed_dict)

                    theta_new_np = self._params.get(as_dict=False)[param_idx].eval(session=sess)

                    Q_rhs = energy_old + np.real(np.sum((theta_new_np - theta_tilde_np) * np.conj(gradient_val))) +\
                                      self._L_np[param_idx][s] / 2.0 * np.real(np.sum((theta_new_np - theta_tilde_np) *
                                                                                      np.conj(theta_new_np - theta_tilde_np)))

                    delta = 1 + np.sign(Q_rhs) * 1e-3
                    if Q_lhs <= Q_rhs * delta:
                        self._L_np[param_idx][s] *= 0.75
                        if self._L_np[param_idx][s] <= 1e-3:
                            self._L_np[param_idx][s] = 1e-3
                        break
                    else:
                        self._L_np[param_idx][s] *= 2.0
                        if self._L_np[param_idx][s] > 1e12:
                            self._L_np[param_idx][s] = 1e12
                            break

                print('*** stage=%d, L=%f, param=%s, Qlhs=%f, Qrhs=%f ||g||=%e' % (s,
                self._L_np[param_idx][s], self._params.get(as_dict=False)[param_idx].name, Q_lhs, Q_rhs,
                                                np.sqrt(np.real(np.sum(gradient_val*np.conj(gradient_val))))))