import tensorflow as tf
import numpy as np
import time
from self_atten_op import *


class SELF_ATTEN_KT():
    def __init__(self, hp, name="selfattenkt"):
        self.hp = hp
        self.name = name

        tag = int(time.time())
        self.config_name = 'logs/%s_%d.config' % (self.name, tag)
        self.log_name = 'logs/%s_%d.log' % (self.name, tag)
        with open(self.config_name, 'w') as f:
            for item in self.hp.__dict__.items():
                f.write(str(item)+'\n')
        
    def creat_graph(self):
        self.tf_x = tf.placeholder(tf.int32, [None, self.hp.maxlen], name='tf_x')
        self.tf_action = tf.placeholder(tf.int32, [None, self.hp.maxlen], name='tf_action') # next action,
        self.tf_y = tf.placeholder(tf.float32, [None, self.hp.maxlen], name='tf_y')
        self.dropout_rate = tf.placeholder(tf.float32, [None], name='tf_dropout')
        self.tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')
        
        # self.tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')
        # self.tf_batch_size = tf.shape(self.tf_x)[0]
        # self.max_seq_len = tf.shape(self.tf_x)[1]

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.tf_x, 0)), -1)

        # (q, a) sequence embedding
        with tf.variable_scope('SAKT', reuse=tf.AUTO_REUSE):
            self.x = embedding(inputs=self.tf_x,
                                vocab_size=self.hp.skill_num*2+1,
                                num_units=self.hp.hidden_units,
                                zero_pad=True,
                                scale=True,
                                l2_reg=self.hp.l2_emb,
                                scope="input_embedding")

            # # Positional Encoding
            # t = embedding(
            #         tf.tile(tf.expand_dims(tf.range(tf.shape(self.tf_x)[1]), 0), [tf.shape(self.tf_x)[0], 1]),
            #         vocab_size=self.hp.maxlen,
            #         num_units=self.hp.hidden_units,
            #         zero_pad=False,
            #         scale=False,
            #         l2_reg=self.hp.l2_emb,
            #         scope="enc_pos")

            # t = positional_encoding(self.x,
            #         zero_pad = False,
            #         scale = False,
            #         scope = "enc_pos")

            t = tf.tile(tf.reshape(tf.range(tf.shape(self.tf_x)[1]), [1,-1,1]), [tf.shape(self.tf_x)[0],1,1])
            self.x += tf.cast(t, tf.float32)
            
            # Dropout
            self.x = tf.layers.dropout(self.x, rate=self.dropout_rate[0])
            self.x *= mask
            self.x = normalize(self.x)

            # build block
            for n in range(self.hp.num_blocks):
                with tf.variable_scope('num_blocks_%d' % n):
                    # self attention
                    self.x, self.weights = multihead_attention(queries=self.x,
                                keys=self.x,
                                values=self.x,
                                num_units=self.hp.hidden_units,
                                num_heads=self.hp.num_heads,
                                dropout_rate=self.dropout_rate[0],
                                causality=True,
                                scope="self_attention")
                    self.x = feedforward(self.x, num_units=[self.hp.hidden_units, self.hp.hidden_units], dropout_rate=self.dropout_rate[0])
                    self.x *= mask

            x_emb = tf.reshape(normalize(self.x), [-1, self.hp.hidden_units])

            with tf.variable_scope('predict'):
                w = tf.get_variable('w', [self.hp.hidden_units, self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('b', [self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
                logits = tf.nn.xw_plus_b(x_emb, w, b)  

        self.all_preds = tf.nn.sigmoid(logits)

        tf_actions_flatten = tf.reshape(self.tf_action, [-1])
        gather_index = tf.transpose(tf.stack([tf.range(tf.shape(logits)[0]), tf_actions_flatten], 0))
        indexed_logits = tf.gather_nd(logits, gather_index)

        tf_targets_flatten = tf.reshape(self.tf_y, [-1])
        index = tf.where(tf.not_equal(tf_targets_flatten, tf.constant(-1, dtype=tf.float32)))
        self.filtered_targets = tf.squeeze(tf.gather(tf_targets_flatten, index), axis=1)
        filtered_logits = tf.squeeze(tf.gather(indexed_logits, index), axis=1)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.filtered_targets,
                                                    logits=filtered_logits))
        self.filtered_predict_prob = tf.nn.sigmoid(filtered_logits)

        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars]) * self.hp.l2_param   # l2_loss /= smaples_number
        self.loss = tf.add(cross_entropy, l2_loss, name='loss')

        # optimize
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


class DKT():
    def __init__(self, hp, name="LSTM"):
        self.hp = hp
        self.name = name

        tag = int(time.time())
        self.config_name = 'logs/%s_%d.config' % (self.name, tag)
        self.log_name = 'logs/%s_%d.log' % (self.name, tag)
        with open(self.config_name, 'w') as f:
            for item in self.hp.__dict__.items():
                f.write(str(item)+'\n')
        
    def creat_graph(self):
        self.tf_x = tf.placeholder(tf.int32, [None, None], name='tf_x')
        self.tf_action = tf.placeholder(tf.int32, [None, None], name='tf_action') # next action,
        self.tf_y = tf.placeholder(tf.float32, [None, None], name='tf_y')
        self.dropout_rate = tf.placeholder(tf.float32, [None], name='tf_dropout')
        self.tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')
        self.tf_batch_size = tf.shape(self.tf_x)[0]
        self.max_seq_len = tf.shape(self.tf_x)[1]

        rnn_inputs = tf.one_hot(self.tf_x, 2*self.hp.skill_num+1, 1., 0.)   # [batch_size, max_seq_len, 2*skill_num+1]
        # with tf.variable_scope('Embedding'):
        #     embed_matrix = tf.get_variable('embed_matrix', [2*self.hp.skill_num+1, self.hp.hidden_units],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
         
        # rnn_inputs = tf.nn.embedding_lookup(embed_matrix, self.tf_x)                    # [batch_size, max_seq_len, embedding_size]


        with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hp.hidden_units, state_is_tuple=True)
            init_state = lstm_cell.zero_state(self.tf_batch_size, tf.float32)
            outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=rnn_inputs,
                                                    sequence_length=self.tf_real_seq_len,
                                                    initial_state=init_state,
                                                    dtype=tf.float32)
            outputs_reshape = tf.reshape(outputs, [-1, self.hp.hidden_units])
           
            rnn_w = tf.get_variable('softmax_w', [self.hp.hidden_units, self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            rnn_b = tf.get_variable('softmax_b', [self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            logits = tf.nn.xw_plus_b(outputs_reshape, rnn_w, rnn_b)  # 
        
        self.all_preds = tf.nn.sigmoid(logits)

        tf_actions_flatten = tf.reshape(self.tf_action, [-1])
        gather_index = tf.transpose(tf.stack([tf.range(tf.shape(logits)[0]), tf_actions_flatten], 0))
        indexed_logits = tf.gather_nd(logits, gather_index)

        tf_targets_flatten = tf.reshape(self.tf_y, [-1])
        index = tf.where(tf.not_equal(tf_targets_flatten, tf.constant(-1, dtype=tf.float32)))
        self.filtered_targets = tf.squeeze(tf.gather(tf_targets_flatten, index), axis=1)
        filtered_logits = tf.squeeze(tf.gather(indexed_logits, index), axis=1)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.filtered_targets,
                                                    logits=filtered_logits))
        self.filtered_predict_prob = tf.nn.sigmoid(filtered_logits)

        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars]) * self.hp.l2_param   # l2_loss /= smaples_number
        self.loss = tf.add(cross_entropy, l2_loss, name='loss')

        # optimize
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)