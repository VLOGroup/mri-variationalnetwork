import tensorflow as tf

class VnBasicData(object):
    def __init__(self, queue_capacity=10):
        # prepare queue
        self._queue_capacity=queue_capacity
        self._tf_dtype = []
        self._queue = None

        self.eval_feed_dict = {}

    def get_feed_dict(self, sess):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    @property
    def tf_dtype(self):
        return self._tf_dtype

    @tf_dtype.setter
    def tf_dtype(self, value):
        if self._queue != None:
            raise ValueError('You are only allowed to set tf_dtype once!')
        self._tf_dtype = value
        self._queue = tf.FIFOQueue(capacity=self._queue_capacity, dtypes=value)
        self._enqueue_op = self._queue.enqueue(self.tf_load())
        self._batch = self._queue.dequeue()

    @property
    def queue(self):
        return self._queue

    @property
    def enqueue_op(self):
        return self._enqueue_op

    def tf_load(self):
        return tf.py_func(self.load, inp=[], Tout=self.tf_dtype, name='vn_data_load')