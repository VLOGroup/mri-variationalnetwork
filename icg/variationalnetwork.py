import tensorflow as tf

class VnBasicCell(object):
    """Variational Network Basic Cell class. The call function defines the calculation for a single stage.

    Args:
        inputs: list of inputs
        constants: dictionary of constants
        params: dictionary of (changing) parameters
        const_params: dictionary of non-changing parameters
        options: dictionary of additional options
    """
    def __init__(self, inputs, params, const_params, constants, options=None):
        self._inputs = inputs
        self._constants = constants
        self._params = params
        self._const_params = const_params
        self._options = options

    @property
    def inputs(self):
        return self._inputs

    @property
    def constants(self):
        return self._constants

    @property
    def params(self):
        return self._params

    def call(self, inputs, t):
        return NotImplementedError('This has to be implemented in the derived class.')

class VariationalNetwork(object):
    """Variational Network class. Defines variational network for a given cell,
     defining a single stage, and a given number of stages.

    Args:
    cell: single stage, defined by the given application
    labels: number of stages
    num_cycles: number of cycles (optional). For all standard variational applications its default value 1 is used.
    parallel_iterations: number of parallel iterations used in while loop (optional) default=1
    swap_memory: Allow swapping of memory in while loop (optional). default=false
    """
    def __init__(self, cell, num_stages, num_cycles=1, parallel_iterations=1, swap_memory=False):
        # Basic computational graph of a vn cell
        self.cell = cell
        # Define the number of repetitions
        self._num_cycles = num_cycles
        self._num_stages = num_stages
        # Tensorflow specific details
        self._parallel_iterations = parallel_iterations
        self._swap_memory = swap_memory

        # Define the iteration method
        def time_to_param_index(t):
            return t % self._num_stages
        self._cell.time_to_param_index = time_to_param_index
        self._t = tf.constant(0)

        # define condition and body for while loop
        self._cond = lambda t, *inputs: t < self._num_stages * self._num_cycles
        def body(t, *inputs):
            cell_outputs = self.cell.call(t, inputs)
            outputs = [tf.concat([i, tf.expand_dims(j, 0)], axis=0) for (i, j) in zip(inputs, cell_outputs)]
            return [t+1] + outputs
        self._body = body

        self._inputs = [tf.expand_dims(inp, 0) for inp in self.cell.inputs]
        self._input_shapes = [tf.TensorShape([None]).concatenate(inp.get_shape()) for inp in self._cell.inputs]
        self._outputs = tf.while_loop(
            self._cond, self._body, loop_vars=[self._t] + self._inputs,
            shape_invariants=[self._t.get_shape()] + self._input_shapes,
            parallel_iterations=self._parallel_iterations,
            swap_memory=self._swap_memory)

    def get_outputs(self, stage_outputs=False):
        """ Get the outputs of the variational network.

        Args:
            stage_outputs: get all stage outputs (optional) default=False
        """
        if stage_outputs:
            return self._outputs[1:]
        else:
            return [out[-1] for out in self._outputs[1:]]

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, value):
        assert isinstance(value, VnBasicCell)
        self._cell = value
