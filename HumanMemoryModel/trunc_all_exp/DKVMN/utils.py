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

def timeJoint(feature, time, proj_length, reg=None):
    with tf.variable_scope('TimeJoint', reuse=tf.AUTO_REUSE, regularizer=reg):
        w_t = tf.get_variable("w_t", [time.shape[1], proj_length],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_t = tf.get_variable("b_t", [proj_length], initializer=tf.constant_initializer(0.1))
        E_s = tf.get_variable("E_s", [proj_length, feature.shape[1]],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

        p_t = tf.nn.xw_plus_b(time, w_t, b_t)
        s_t = tf.nn.softmax(p_t)
        g_t = tf.matmul(s_t, E_s)
        feature = tf.add(feature, g_t) / 2.0
    return feature


def timeMask(feature, time, cont_length, reg=None):
    with tf.variable_scope('TimeMask', reuse=tf.AUTO_REUSE, regularizer=reg):
        w = tf.get_variable("weight", [time.shape[1], cont_length],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias", [cont_length], initializer=tf.constant_initializer(0.1))

        w_t = tf.get_variable("w_t", [cont_length, feature.shape[1]],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_t = tf.get_variable("b_t", [feature.shape[1]], initializer=tf.constant_initializer(0.1))

        c_t = tf.nn.relu(tf.nn.xw_plus_b(tf.log(1+time), w, b))
        # m_t = tf.nn.sigmoid(tf.nn.xw_plus_b(c_t, w_t, b_t))
        m_t = tf.nn.relu(tf.nn.xw_plus_b(c_t, w_t, b_t))
        feature = tf.multiply(feature, m_t)
    return feature