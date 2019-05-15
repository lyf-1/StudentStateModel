import numpy as np
import tensorflow as tf
import pickle as pkl


def linear(x, state_dim, name='linear', reuse=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        weight = tf.get_variable('weight', [x.get_shape()[-1], state_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [state_dim], initializer=tf.constant_initializer(0))
        weighted_sum = tf.nn.xw_plus_b(x, weight, bias)
        return weighted_sum

