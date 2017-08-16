from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class AnswerCell(RNNCell):
    """The modified GRU Cell that serves as the answer module of the dynamic memory network"""

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 vocab_size=0):
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._vocab_size = vocab_size
        #self.state_size = (self._num_units, self._vocab_size)

    @property
    def state_size(self):
        #return self._num_units
        return (self._num_units, self._vocab_size)

    @property
    def output_size(self):
        #return self._num_units
        return self._vocab_size

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "answer_cell"):
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.

                #state, prev_output = array_ops.split(state, num_or_size_splits=2, axis=1)
                # state, prev_output = tf.split(state, [self._num_units, self._vocab_size], axis=1)
                state, prev_output = state
                # inputs = [inputs, prev_output]
                inputs = tf.concat([inputs, prev_output], 1)
                bias_ones = self._bias_initializer
                if self._bias_initializer is None:
                    dtype = [a.dtype for a in [inputs, state]][0]
                    bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
                value = math_ops.sigmoid(
                    _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                            self._kernel_initializer))
                r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
            with vs.variable_scope("candidate"):
                c = self._activation(
                    _linear([inputs, r * state], self._num_units, True,
                            self._bias_initializer, self._kernel_initializer))
            new_h = u * state + (1 - u) * c
            logits_y = _linear(new_h, self._vocab_size, bias_ones,
                               self._kernel_initializer)
            softmax_y = tf.nn.softmax(logits_y)
            # new_state = tf.concat([new_h, softmax_y], 1)
            new_state = (new_h, softmax_y)
            #print('new_state size')
            #print(new_state.get_shape())
            return logits_y, new_state


def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)