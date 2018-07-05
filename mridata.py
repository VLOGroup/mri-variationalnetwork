import tensorflow as tf
import numpy as np
import mriutils
import glob
import vn
import time
import os

import tensorflow.contrib.icg as icg

class VnMriFilenameProducer(object):
    """Filename Producer Class for MRI data. Based on the config, it generates a list of all files that should be used
    for training and randomly shuffles the data and stores them in a queue. It ensures that all data is seen once in an
    epoch. For dequeing, a batch of filenames is taken out of the queue, defined by the config parameter `batch_size`.
    All filenames of one batch belong to the same dataset. This allows for training with datasets of different image
    dimension, because it is ensured that all data in the batch have the same image dimension.

    Args:
    config: Data config
    labels: number of stages
    queue_capacity: number of filename lists holding all file names that are stored in a queue
    """
    def __init__(self, config, queue_capacity = 2):
        # store the data config
        self.config = config

        self.slice_idx_list = []
        self.patient_idx_list = []

        # generate all filenames
        for dataset_idx in range(0, len(config['dataset'])):
            path = os.path.expanduser(config['base_dir'] + '/' + config['dataset'][dataset_idx]['name'] + '/')
            dataset_patient = []
            dataset_slice = []
            for patient_idx in config['dataset'][dataset_idx]['patients']:
                num_slices = len(glob.glob(path + '/%d/rawdata*.mat' % (patient_idx)))
                end_slice = config['dataset'][dataset_idx]['end_slice']
                if end_slice == None or end_slice > num_slices:
                    end_slice = num_slices
                for slice_idx in range(config['dataset'][dataset_idx]['start_slice'], end_slice+1):
                    dataset_patient.append(patient_idx)
                    dataset_slice.append(slice_idx)
            self.patient_idx_list.append(dataset_patient)
            self.slice_idx_list.append(dataset_slice)

        self.iter_per_epoch = 0
        for d in range(len(self.config['dataset'])):
            self.iter_per_epoch += int(len(self.patient_idx_list[d])/self.config['batch_size'])

        # dequeue operation returns int64 to identify [dataset, patient, slice] that should be loaded
        self.tf_dtype = [tf.int64, tf.int64, tf.int64]
        # setup queue
        self.queue = tf.FIFOQueue(capacity=queue_capacity, dtypes = self.tf_dtype, shapes=[[],[],[]])
        self.enqueue_op = self.queue.enqueue_many(self.tf_load())
        self.dequeue_op = self.queue.dequeue_many(self.config['batch_size'])

    def tf_load(self):
        return tf.py_func(self.load, inp=[], Tout=self.tf_dtype)

    def load(self):
        # permute over all individual datasets
        slice_idx_list = []
        patient_idx_list = []
        append_batches = []
        for d in range(len(self.config['dataset'])):
            num_files = len(self.patient_idx_list[d])
            permuted_idx = np.random.permutation(list(range(0, num_files)))
            slice_idx_list.append(list(self.slice_idx_list[d][idx] for idx in permuted_idx))
            patient_idx_list.append(list(self.patient_idx_list[d][idx] for idx in permuted_idx))
            append_batches.append(True)

        shuffeled_slice_idx_list = []
        shuffeled_patient_idx_list = []
        shuffeled_dataset_idx_list = []

        b = 0

        # make sure that the appended batches have always the size `batch_size`
        # randomly shuffle datasets, patients and slices
        while True in append_batches:
            permuted_datasets = list(np.random.permutation(list(range(0, len(self.config['dataset'])))))
            for d in permuted_datasets:
                if len(patient_idx_list[d][b:b+self.config['batch_size']]) < self.config['batch_size']:
                    append_batches[d] = False
                else:
                    shuffeled_dataset_idx_list.extend([d for i in range(self.config['batch_size'])])
                    shuffeled_patient_idx_list.extend(patient_idx_list[d][b:b+self.config['batch_size']])
                    shuffeled_slice_idx_list.extend(slice_idx_list[d][b:b+self.config['batch_size']])
            b+=self.config['batch_size']
        return [shuffeled_dataset_idx_list, shuffeled_patient_idx_list, shuffeled_slice_idx_list]

class VnMriReconstructionData(vn.VnBasicData):
    """Data class for loading MRI data.

    Args:
    config: Data config
    filename_dequeue_op: Dequeue op to get next batch of file names that should be loaded
    queue_capacity: Number of batches that can be stored in a queue
    u_var: tf variable holding zero filled reconstruction. Can be used when graph is loaded / evaluation
    f_var: tf variable holding k-space data. Can be used when graph is loaded / evaluation
    g_var: tf variable holding reference reconstruction. Can be used when graph is loaded / evaluation
    c_var: tf variable holding coil sensitivity maps. Can be used when graph is loaded / evaluation
    m_var: tf variable holding the sampling mask.
    load_target: boolean to define if reference reconstruction should be loaded.
    load_eval_data: boolean to define if evaluation data should be loaded. This should be true for training.
    """
    def __init__(self, config, filename_dequeue_op=[], queue_capacity=10,
                 u_var=None, f_var=None, g_var=None, c_var=None, m_var=None,
                 load_target=True, load_eval_data=True):

        self.data_config = config
        self.load_target = load_target

        super(VnMriReconstructionData, self).__init__(queue_capacity=queue_capacity)

        self.filename_dequeue_op = filename_dequeue_op

        if load_target:
            assert config['batch_size'] >= 1 and isinstance(config['batch_size'], int)
            self._batch_size = config['batch_size']

        # override tf_load function *before* dtype is set!
        def tf_load():
            return tf.py_func(self.load, inp=self.filename_dequeue_op, Tout=self.tf_dtype)
        self.tf_load = tf_load

        # all tf variables except the sampling mask are complex-valued
        self.tf_dtype = [tf.complex64 for i in range(4)] + [tf.float32]

        if u_var!=None and f_var!=None and g_var!=None and m_var != None and c_var != None:
            vars_defined = True
        elif u_var!=None or f_var!=None or g_var!=None or m_var != None or c_var != None:
            raise ValueError('Only some variables of the graph were defined!')
        else:
            vars_defined = False

        if vars_defined:
            self.u = u_var[0]
            self.target = g_var[0]
            self.constants = {'f' : f_var[0],
                              'coil_sens' : c_var[0],
                              'sampling_mask' : m_var[0]
                              }
        else:
            # define inputs
            self.u = tf.placeholder(shape=(None, None, None), dtype=tf.complex64, name='u')

            # define constants
            self.constants = {'f': tf.placeholder(shape=(None, None, None, None), dtype=tf.complex64, name='f'),
                              'coil_sens': tf.placeholder(shape=(None, None, None, None), dtype=tf.complex64,
                                                          name='coil_sens'),
                              'sampling_mask': tf.placeholder(shape=(None, None, None), dtype=tf.float32,
                                                              name='sampling_mask')}

            # define the target
            self.target = tf.placeholder(shape=(None, None, None), dtype=tf.complex64, name='g')

        if load_eval_data:
            # create eval feed dict
            self.eval_data = {'f' : [], 'coil_sens' : [], 'u' : [], 'sampling_mask' : [], 'cost_mask' : [], 'g' : []}
            for didx in range(0, len(config['dataset'])):
                for pidx in config['dataset'][didx]['eval_patients']:
                    for sidx in config['dataset'][didx]['eval_slices']:
                        f, c, input0, ref, mask, _ = self._load_single(self.data_config['dataset'][didx], pidx, sidx)
                        self.eval_data['f'].append(f)
                        self.eval_data['coil_sens'].append(c)
                        self.eval_data['sampling_mask'].append(mask)
                        self.eval_data['g'].append(ref)
                        self.eval_data['u'].append(input0)

    def get_feed_dict(self, sess):
        """ Get feed dictionary for training."""
        [input0_batch, kspace_batch, coil_sens_batch, reference_batch, sampling_mask_batch] = sess.run(self._batch)

        return {self.u: input0_batch,
            self.constants['f']: kspace_batch,
            self.constants['coil_sens']: coil_sens_batch,
            self.constants['sampling_mask']: sampling_mask_batch,
            self.target: reference_batch,
            }

    def get_eval_feed_dict(self):
        """ Get feed dictionary for evaluation. This does not change during training!"""
        feed_dict = {self.target: np.array(self.eval_data['g'][0:0+self._batch_size:]),
                     self.u: np.array(self.eval_data['u'][0:0+self._batch_size:]),
                    }
        for key in self.constants.keys():
            feed_dict[self.constants[key]] = np.array(self.eval_data[key][0:0+self._batch_size:])

        return feed_dict

    def get_test_feed_dict(self, dataset, patient, slice, return_norm=False):
        """ Get feed dictionary for testing a specific dataset, patient and slice.

        Args:
        dataset: int defining the dataset in the data config
        patient: int defining the patient
        slice: int defining the slice
        return_norm: data is normalized slice-per-slice. To be able to re-normalize, set return_norm=True
        """
        kspace, coil_sens, input0, ref, sampling_mask, norm = self._load_single(dataset, patient, slice)


        feed_dict = { self.u: np.asarray([input0]),
                      self.target: np.asarray([ref]),
                      self.constants['f']: np.asarray([kspace]),
                        self.constants['coil_sens']: np.asarray([coil_sens]),
                        self.constants['sampling_mask']: np.asarray([sampling_mask]),
                    }

        if return_norm:
            return feed_dict, norm
        else:
            return feed_dict


    def _load_single(self, dataset, patient, idx):
        """ Get feed dictionary for testing a specific dataset, patient and slice.

        Args:
        dataset: int defining the dataset in the data config
        patient: int defining the patient
        slice: int defining the slice
        """

        # extract paths
        path = os.path.expanduser(self.data_config['base_dir'] + '/' + dataset['name'] + '/')

        def load_mat():
            """
            Load the matlab data.
            """
            # sampling mask path
            mask_path = '%s/%s' % (path, dataset['mask'])

            # load mask
            mask_matlab_data = icg.utils.loadmat(mask_path)
            mask = mask_matlab_data['mask'].astype(np.float32)

            # load matlab raw data and sensitivity maps. The type of sensitivities is defined in `sens_type` of the data config.
            matlab_rawdata = icg.utils.loadmat(path + '/%d/rawdata%d.mat' % (patient, idx))
            matlab_sens = icg.utils.loadmat(path + '/%d/%s%d.mat' % (patient, self.data_config['sens_type'], idx))

            # re-organize data
            f = np.ascontiguousarray(np.transpose(matlab_rawdata['rawdata'], (2, 0, 1))).astype(np.complex64)
            c = np.ascontiguousarray(np.transpose(matlab_sens['sensitivities'], (2, 0, 1))).astype(np.complex64)

            # load reference
            if self.load_target:
                ref = matlab_sens['reference'].astype(np.complex64)
            else:
                ref = np.zeros_like(mask, dtype=np.complex64)

            # padlength variables to define the number of lines for phase encoding oversampling.
            if 'padlength_left' in matlab_rawdata and 'padlength_right' in matlab_rawdata:
                padlength_left = int(matlab_rawdata['padlength_left'])
                padlength_right = int(matlab_rawdata['padlength_right'])
            else:
                padlength_left = 0
                padlength_right = 0

            return mask, f, c, ref, padlength_left, padlength_right


        mask, f, c, ref, padlength_left, padlength_right = load_mat()

        # pad mask with ones to ensure that the reconstruction is forced to zero in the phase-encoding oversampled region.
        if padlength_left > 0:
            mask[:,:padlength_left] = 1
        if padlength_right > 0:
            mask[:,-padlength_right:] = 1

        # mask rawdata
        f *= mask

        # compute initial image input0
        input0 = mriutils.mriAdjointOp(f, c, mask).astype(np.complex64)

        # remove frequency encoding oversampling
        if self.data_config['sampling_pattern'] == 'cartesian_with_os':
            if self.load_target:
                ref = mriutils.removeFEOversampling(ref)  # remove RO Oversampling
            input0 = mriutils.removeFEOversampling(input0)  # remove RO Oversampling
        elif self.data_config['sampling_pattern'] == 'cartesian':
            pass
        else:
            raise ValueError("'sampling_pattern' has to be in [cartesian_with_os, cartesian]")

        # normalize the data
        # To streamline the implementation we normalize by the max value of the zero-filled recon
        # This is different to the paper.
        if self.data_config['normalization'] == 'max':
            norm = np.max(np.abs(input0))
        elif self.data_config['normalization'] == 'no':
            norm = 1.0
        else:
            raise ValueError("Normalization has to be in ['max', 'no']")

        f /= norm
        input0 /= norm

        if self.load_target:
            ref /= norm
        else:
            ref = np.zeros_like(input0)

        return f, c, input0, ref, mask, norm

    def load(self, dataset_batch, patient_batch, slice_batch):
        """ Load batch data..

        Args:
        dataset_batch: list of ints defining the dataset in the data config
        patient_batch: list of ints defining the patient
        slice_batch: list of ints int defining the slice
        """
        t = time.time()
        # generate batch
        input0_batch = []
        coil_sens_batch = []
        kspace_batch = []
        sampling_mask_batch = []
        reference_batch = []

        for i in range(self._batch_size):
            kspace, coil_sens, input0, reference, sampling_mask, _ = self._load_single(self.data_config['dataset'][dataset_batch[i]],
                                                                                       patient_batch[i],
                                                                                       slice_batch[i])
            input0_batch.append(input0)
            kspace_batch.append(kspace)
            coil_sens_batch.append(coil_sens)
            reference_batch.append(reference)
            sampling_mask_batch.append(sampling_mask)

        print('[DataThread] Loading took ', time.time() - t)

        return [np.asarray(input0_batch),
                np.asarray(kspace_batch),
                np.asarray(coil_sens_batch),
                np.asarray(reference_batch),
                np.asarray(sampling_mask_batch)]