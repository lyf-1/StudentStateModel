import tensorflow as tf
import numpy as np
import pickle as pkl


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


def lstm_net(tf_features, tf_seq_len, tf_actions_flatten, init_c, init_h, n_hidden, n_items):
    with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
        lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
        init_state = tf.contrib.rnn.LSTMStateTuple(init_c, init_h)
        outputs, last_state = tf.nn.dynamic_rnn(cell=lstm, 
                            inputs=tf_features, sequence_length=tf_seq_len, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden])
        softmax_w = tf.get_variable('softmax_w', [n_hidden, n_items+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
        softmax_b = tf.get_variable('softmax_b', [n_items+1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)     
        gather_index = tf.transpose(tf.stack([tf.range(tf.shape(logits)[0]), tf_actions_flatten], 0))
        indexed_logits = tf.gather_nd(logits, gather_index)
        return last_state, indexed_logits


def split_data(datalist, rate=0.8):
    splitpoint = int(datalist[0].shape[0] * 0.8)
    splitdata = []
    for data in datalist:
        train = data[:splitpoint, :]
        valid = data[splitpoint:, :]
        splitdata.append(train)
        splitdata.append(valid)
    return splitdata
    

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]


def batch_generator(data, batch_size_, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size_ >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size_
        end = start + batch_size_
        batch_count += 1
        yield [d[start:end] for d in data]


def pass_rate(qa_data, n):
    ans = (qa_data-1) // n
    num_pass = np.sum(ans==1)
    num_fail = np.sum(ans==0)
    print('num pass / num fail', num_pass, num_fail)
    return float(num_pass) / (num_fail+num_pass)
