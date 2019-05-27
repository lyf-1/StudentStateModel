from tensorflow.python.eager import context
from tensorflow.python.ops.rnn_cell_impl import RNNCell, LSTMStateTuple
from tensorflow.python.framework import constant_op, dtypes
from tensorflow.python.ops import array_ops, math_ops, nn_ops, init_ops
from tensorflow.python.platform import tf_logging as logging
from linear import _linear


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class TimeLstmCell(RNNCell):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    
    super(TimeLstmCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    # if context.executing_eagerly() and context.num_gpus() > 0:
    #   logging.warn("%s: Note that this cell is not optimized for performance. "
    #                "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
    #                "performance on GPU.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  def call(self, inputs, state):
    inputs, time = inputs[0], inputs[1]  # time, 2-d matrix 

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    sigmoid = math_ops.sigmoid
    add = math_ops.add
    multiply = math_ops.multiply
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    # input_gate, new_input, forget_gate
    gate_inputs = _linear([inputs, h], 
                          3 * self._num_units, 
                          bias=True,
                          weight_name='weight_ijf',
                          bias_name='bias_ijf')
    i, j, f = array_ops.split(
        value=gate_inputs, num_or_size_splits=3, axis=one)

    # time_gate
    time_gate_tt = _linear([time],  
                            self._num_units,
                            bias=False,
                            weight_name='weight_tt')
    
    time_gate_xt = _linear([inputs],  
                            self._num_units,
                            bias=True,
                            weight_name='weight_xt',
                            bias_name='bias_xt')
    t = add(time_gate_xt, sigmoid(time_gate_tt)) 

    # output_gate 
    o = _linear([inputs, h, time],
                 self._num_units,
                 bias=True,
                 weight_name='weight_o',
                 bias_name='bias_o')
    
    # update
    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                multiply(multiply(sigmoid(i), self._activation(j)), sigmoid(t)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state


class TimeLstmCell3(RNNCell):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    
    super(TimeLstmCell3, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    # if context.executing_eagerly() and context.num_gpus() > 0:
    #   logging.warn("%s: Note that this cell is not optimized for performance. "
    #                "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
    #                "performance on GPU.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  def call(self, inputs, state):
    inputs, time = inputs[0], inputs[1]  # time, 2-d matrix 

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    sigmoid = math_ops.sigmoid
    add = math_ops.add
    multiply = math_ops.multiply
    one = constant_op.constant(1, dtype=dtypes.int32)
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    # input_gate, new_input
    gate_inputs = _linear([inputs, h], 
                          2 * self._num_units, 
                          bias=True,
                          weight_name='weight_ijf',
                          bias_name='bias_ijf')
    i, j = array_ops.split(
        value=gate_inputs, num_or_size_splits=2, axis=one)

    # time_gate1
    time_gate_tt1 = _linear([time],  
                            self._num_units,
                            bias=False,
                            weight_name='weight_tt1',
                            clip_weights=True)
    
    time_gate_xt1 = _linear([inputs],  
                            self._num_units,
                            bias=True,
                            weight_name='weight_xt1',
                            bias_name='bias_xt1')
    t1 = add(time_gate_xt1, sigmoid(time_gate_tt1)) 

    # time_gate2
    time_gate_tt2 = _linear([time],  
                        self._num_units,
                        bias=False,
                        weight_name='weight_tt2')
    
    time_gate_xt2 = _linear([inputs],  
                            self._num_units,
                            bias=True,
                            weight_name='weight_xt2',
                            bias_name='bias_xt2')
    t2 = add(time_gate_xt2, sigmoid(time_gate_tt2)) 

    # output_gate 
    o = _linear([inputs, h, time],
                 self._num_units,
                 bias=True,
                 weight_name='weight_o',
                 bias_name='bias_o')

    # couple input and forget gates
    one_float32 = constant_op.constant(1, dtype=dtypes.float32)
    f1 = add(one_float32, multiply(-sigmoid(i), sigmoid(t1)))
    f2 = add(one_float32, -sigmoid(i))

    # update
    c_t = add(multiply(c, f1), multiply(multiply(sigmoid(i), self._activation(j)), sigmoid(t1)))
    new_c = add(multiply(c, f2), multiply(multiply(sigmoid(i), self._activation(j)), sigmoid(t2)))
    new_h = multiply(self._activation(c_t), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state
