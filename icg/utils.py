import tensorflow as tf
import os
from scipy.io import loadmat as scipy_loadmat
import h5py
import numpy as np
import yaml
import re
import shutil

def log10(x):
    """ Compute log10 with tf.
    """
    num = tf.log(x)
    den = tf.log(tf.constant(10, dtype=num.dtype))
    return num / den

def loadmat(filename, variable_names=None):
    """ load mat file independent of version it is saved with.
    """
    success = False
    try:
        data = __load_mat_below_7_3(filename, variable_names)
        success = True
    except Exception as e:
        #print e
        pass
        #print 'Failed reading mat file using scipy.io.loadmat. Try h5py to load file %s' % filename

    if not success:
        try:
            data = __load_mat_7_3(filename, variable_names)
            success = True
        except Exception as e:
            #print e
            pass
            #print 'Failed to load file %s using h5py.' % filename

    if success:
        return data
    else:
        raise ValueError('Matlab file could not be read! Invalid filepath?')

def __load_mat_below_7_3(filename, variable_names=None):
    matfile = scipy_loadmat(filename, variable_names=variable_names)
    data = {}
    for key in matfile.keys():
        if isinstance(matfile[key], str) or  \
           isinstance(matfile[key], list) or \
           isinstance(matfile[key], dict) or \
           key == '__header__' or key == '__globals__' or key == '__version__':
            data.update({key: matfile[key]})
        elif  matfile[key].dtype.names != None and 'imag' in matfile[key].dtype.names:
            data.update({key: np.asarray(matfile[key].real + 1j*matfile[key].imag, dtype='complex128')})
        else:
            data.update({key: np.asarray(matfile[key], dtype=matfile[key].dtype)})
    return data

def __load_mat_7_3(filename, variable_names=None):
    matfile = h5py.File(filename, 'r')
    data = {}
    if  variable_names == None:
        for key in matfile.keys():
            if isinstance(matfile[key], str) or  \
               isinstance(matfile[key], list) or \
               isinstance(matfile[key], dict) or \
               key == '__header__' or key == '__globals__' or key == '__version__':
                data.update({key: matfile[key]})
            elif  matfile[key].dtype.names != None and 'imag' in matfile[key].dtype.names:
                data.update({key: np.transpose(np.asarray(matfile[key].value.view(np.complex), dtype='complex128'))})
            else:
                data.update({key: np.transpose(np.asarray(matfile[key].value, dtype=matfile[key].dtype))})
    else:
        for key in variable_names:
            if not key in matfile.keys():
                raise RuntimeError('Variable: "' + key + '" is not in file: '+ filename)
            if isinstance(matfile[key], str) or  \
               isinstance(matfile[key], list) or \
               isinstance(matfile[key], dict) or \
               key == '__header__' or key == '__globals__' or key == '__version__':
                data.update({key: matfile[key]})
            elif  matfile[key].dtype.names != None and 'imag' in matfile[key].dtype.names:
                data.update({key: np.transpose(np.asarray(matfile[key].value.view(np.complex), dtype='complex128'))})
            else:
                data.update({key: np.transpose(np.asarray(matfile[key].value, dtype=matfile[key].dtype))})

    return data

def loadYaml(config_path, keys):
    """ load yaml configuration file for specific keys.
    """
    # Parse environment variables
    # see: https://stackoverflow.com/questions/26712003/pyyaml-parsing-of-the-environment-variable-in-the-yaml-configuration-file
    # define the regex pattern that the parser will use to 'implicitely' tag your node
    pattern = re.compile(r'^\<%= ENV\[\'(.*)\'\] %\>(.*)$')

    # now define a custom tag ( say pathex ) and associate the regex pattern we defined
    yaml.add_implicit_resolver("!pathex", pattern)

    # at this point the parser will associate '!pathex' tag whenever the node matches the pattern

    # you need to now define a constructor that the parser will invoke
    # you can do whatever you want with the node value
    def pathex_constructor(loader, node):
        value = loader.construct_scalar(node)
        envVar, remainingPath = pattern.match(value).groups()
        return os.environ[envVar] + remainingPath

    # 'register' the constructor so that the parser will invoke 'pathex_constructor' for each node '!pathex'
    yaml.add_constructor('!pathex', pathex_constructor)

    # load the configs
    configs = []
    with open(config_path, 'r') as stream:
        try:
            cfg = yaml.load(stream)
            [configs.append(cfg[key]) for key in keys]
        except yaml.YAMLError as exc:
            print(exc)
    if len(configs) == 1:
        return configs[0]
    else:
        return tuple(configs)

class Params(object):
    """ Parameter class holding tensorflow variables used for custom icg optimizers (IPALM) and variational network. 
        Allows to hold a prox operator, defined in either tensorflow or numpy, for each parameter
        as well as a step size tau.
    """
    def __init__(self):
        self._tf_params_dict = {}
        self._tf_params_list = []
        self._tf_prox_list = []
        self._tf_tau_list = []
        self._np_prox_list = []

    def add(self, tf_var, prox=None, tau=None, np_prox=None):
        var_name = tf_var.name.split(':')[0]
        self._tf_params_dict.update({var_name : tf_var})
        self._tf_params_list.append(tf_var)
        self._tf_prox_list.append(prox)
        self._tf_tau_list.append(tau)
        self._np_prox_list.append(np_prox)

        if prox is not None and np_prox is not None:
            raise RuntimeError('Just a single prox is supported! Use either prox or np_prox!')

    def _get_output_dict(self, output_list):
        output_dict = {}
        assert len(output_list) == len(self._tf_params_list)
        for idx in range(len(output_list)):
            var_name = self._tf_params_list[idx].name.split(':')[0]
            output_dict.update({var_name : output_list[idx]})
        return output_dict

    def get(self, as_dict=True):
        if as_dict:
            return self._tf_params_dict
        else:
            return self._tf_params_list

    def get_prox(self):
        return self._tf_prox_list

    def get_tau(self):
        return self._tf_tau_list

    def get_np_prox(self):
        return self._np_prox_list

    def eval(self, sess, name=None, as_dict = True):
        if name != None:
            return sess.run(self._tf_params_dict[name])
        else:
            output_list = sess.run(self._tf_params_list)
            if as_dict:
                return self._get_output_dict(output_list)
            else:
                return output_list

class ConstParams(object):
    """ Constant (non-changing) parameter class holding tensorflow variables used for custom icg
        optimizers (IPALM) and variational network.
    """
    def __init__(self):
        self._tf_params_dict = {}
        self._tf_params_list = []

    def add(self, tf_var):
        var_name = tf_var.name.split(':')[0]
        self._tf_params_dict.update({var_name : tf_var})
        self._tf_params_list.append(tf_var)

    def get(self, as_dict = True):
        if as_dict:
            return self._tf_params_dict
        else:
            return self._tf_params_list

    def eval(self, sess, name=None, as_dict = True):
        if name != None:
            return sess.run(self._tf_params_dict[name])
        else:
            output_list = sess.run(self._tf_params_list)
            if as_dict:
                return self._get_output_dict(output_list)
            else:
                return output_list

