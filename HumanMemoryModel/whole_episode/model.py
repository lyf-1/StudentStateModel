import tensorflow as tf
from utils import timeJoint, timeMask, linear
from dkvmn_memory import DKVMN
import time


# class EFC():
#     def __init__(self, hp, name="LSTM"):
#         self.hp = hp
#         self.name = name

#         tag = int(time.time())
#         self.config_name = 'logs/%s_%d.config' % (self.name, tag)
#         self.log_name = 'logs/%s_%d.log' % (self.name, tag)
#         with open(self.config_name, 'w') as f:
#             for item in self.hp.__dict__.items():
#                 f.write(str(item)+'\n')
        
#     def creat_graph(self):
#         # placeholder
#         self.tf_x = tf.placeholder(tf.int32, [None, None], name='tf_x')
#         self.tf_action = tf.placeholder(tf.int32, [None, None], name='tf_action') # next action,
#         self.tf_y = tf.placeholder(tf.float32, [None, None], name='tf_y')
#         self.tf_time = tf.placeholder(tf.float32, [None, None])    # tlast
#         self.tf_nreps = tf.placeholder(tf.float32, [None, None])    # nreps
#         self.tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')
#         self.tf_batch_size = tf.shape(self.tf_x)[0]

#         # build model 
#         reg = tf.contrib.layers.l2_regularizer(scale=self.hp.l2_param)
#         with tf.variable_scope('efc', reuse=tf.AUTO_REUSE, regularizer=reg):
#             tf_item_difficulties = tf.get_variable('item_difficulties', [self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
#         item_diff = tf.gather(tf_item_difficulties, tf_item_idx)
#         pred_precall = tf.exp(-item_diff*(tf_tlast/tf_nreps))
#         pred_precall = tf.clip_by_value(pred_precall, .0001, .9999)
    

#         precall_loss = -tf.reduce_mean(tf_targets*tf.log(pred_precall)+(1-tf_targets)*tf.log(1-pred_precall)) # cross_entropy
#         l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#         self.loss = precall_loss + l2_loss

#         # optimize
#         self.global_step = tf.Variable(0, name='global_step', trainable=False)
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.lr, beta2=0.98)
#         self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)



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
        # placeholder
        self.tf_x = tf.placeholder(tf.int32, [None, None], name='tf_x')
        self.tf_action = tf.placeholder(tf.int32, [None, None], name='tf_action') # next action,
        self.tf_y = tf.placeholder(tf.float32, [None, None], name='tf_y')
        self.tf_time = tf.placeholder(tf.float32, [None, None])    # tlast
        self.tf_nreps = tf.placeholder(tf.float32, [None, None])    # nreps
        self.tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')
        self.tf_batch_size = tf.shape(self.tf_x)[0]
        self.max_seq_len = tf.shape(self.tf_x)[1]

        # feature embedding
        with tf.variable_scope('Embedding'):
            embed_matrix = tf.get_variable('embed_matrix', [2*self.hp.skill_num+1, self.hp.embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
         
        rnn_inputs = tf.nn.embedding_lookup(embed_matrix, self.tf_x)                    # [batch_size, max_seq_len, embedding_size]
        rnn_inputs_reshape = tf.reshape(rnn_inputs, [-1, self.hp.embedding_size])

        # add: decay_factor = [time, nreps]
        tf_time_reshape = tf.reshape(self.tf_time, [-1, 1])
        tf_nreps_reshape = tf.reshape(self.tf_nreps, [-1, 1])
        tf_decay_factor = tf.concat([tf_time_reshape, tf_nreps_reshape], axis=1)
        feature_dim = self.hp.embedding_size
        if self.hp.decay_factor_in_input == 'timeConcate':
            print('timeConcate')
            rnn_inputs_with_decay = tf.concat([rnn_inputs_reshape, tf_decay_factor], axis=1)
            feature_dim += 2
        elif self.hp.decay_factor_in_input == 'timeMask':
            print('timeMask')
            rnn_inputs_with_decay = timeMask(rnn_inputs_reshape, tf_decay_factor, self.hp.proj_len)
        elif self.hp.decay_factor_in_input == 'timeJoint':
            print('timeJoint')
            rnn_inputs_with_decay = timeJoint(rnn_inputs_reshape, tf_decay_factor, self.hp.proj_len)
        else:
            print('notime')
            rnn_inputs_with_decay = rnn_inputs_reshape
        rnn_inputs_with_decay = tf.reshape(rnn_inputs_with_decay, [-1, self.max_seq_len, feature_dim])


        with tf.variable_scope('LstmNet', reuse=tf.AUTO_REUSE):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hp.hidden_units, state_is_tuple=True)
            init_state = lstm_cell.zero_state(self.tf_batch_size, tf.float32)
            outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=rnn_inputs_with_decay,
                                                    sequence_length=self.tf_real_seq_len,
                                                    initial_state=init_state,
                                                    dtype=tf.float32)
            outputs_reshape = tf.reshape(outputs, [-1, self.hp.hidden_units])
           
            rnn_w = tf.get_variable('softmax_w', [self.hp.hidden_units, self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            rnn_b = tf.get_variable('softmax_b', [self.hp.skill_num+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            logits = tf.nn.xw_plus_b(outputs_reshape, rnn_w, rnn_b)  # 
        

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



class DKVMN_model():
    def __init__(self, hp, name='DKVMN'):
        self.args = hp
        self.name = name

        tag = int(time.time())
        self.config_name = 'logs/%s_%d.config' % (self.name, tag)
        self.log_name = 'logs/%s_%d.log' % (self.name, tag)
        with open(self.config_name, 'w') as f:
            for item in self.args.__dict__.items():
                f.write(str(item)+'\n')

    def creat_graph(self):
        self.tf_action = tf.placeholder(tf.int32, [None, self.args.maxlen], name='q_data')
        self.tf_x = tf.placeholder(tf.int32, [None, self.args.maxlen], name='qa_data')
        self.tf_y = tf.placeholder(tf.float32, [None, self.args.maxlen], name='target')
        self.tf_time = tf.placeholder(tf.float32, [None, self.args.maxlen], name='t_data')
        self.tf_nreps = tf.placeholder(tf.float32, [None, self.args.maxlen], name='nreps_data')
        self.tf_real_seq_len = tf.placeholder(tf.int32, [None], name='tf_real_seq_len')  # no use

        with tf.variable_scope('Memory'):
            init_key_memory = tf.get_variable('key', [self.args.mem_size, self.args.key_mem_state_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            init_value_memory = tf.get_variable('value', [self.args.mem_size, self.args.value_mem_state_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        init_value_memory = tf.tile(tf.expand_dims(init_value_memory, axis=0), tf.stack([tf.shape(self.tf_action)[0], 1, 1]))

        self.memory = DKVMN(self.args.mem_size, self.args.key_mem_state_dim, self.args.value_mem_state_dim, 
                            init_key_memory, init_value_memory, name='DKVMN')
        
        with tf.variable_scope('Embedding'):
            q_embed_mtx = tf.get_variable('q_embed', [self.args.skill_num+1, self.args.key_mem_state_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.skill_num+1, self.args.value_mem_state_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        q_embed_data = tf.nn.embedding_lookup(q_embed_mtx, self.tf_action)        # [batch_size, seq_len, key_mem_state_dim]
        slice_q_embed_data = tf.split(q_embed_data, self.args.maxlen, 1)  
        slice_t_data = tf.split(self.tf_time, self.args.maxlen, 1)
        slice_nreps_data = tf.split(self.tf_nreps, self.args.maxlen, 1)

        qa_embed_data = tf.nn.embedding_lookup(qa_embed_mtx, self.tf_x)     # [batch_size, seq_len, value_mem_state_dim]      
        qa_embed_reshape = tf.reshape(qa_embed_data, [-1, self.args.value_mem_state_dim])
        # add decay_factor = [time, nreps] in input layer
        tf_time_reshape = tf.reshape(self.tf_time, [-1, 1])
        tf_nreps_reshape = tf.reshape(self.tf_nreps, [-1, 1])
        tf_decay_factor = tf.concat([tf_time_reshape, tf_nreps_reshape], axis=1)
        feature_dim = self.args.value_mem_state_dim
        if self.args.decay_factor_in_input == 'timeConcate':
            print('timeConcate')
            qa_embed_with_decay = tf.concat([qa_embed_reshape, tf_decay_factor], axis=1)
            feature_dim += 2
        elif self.args.decay_factor_in_input == 'timeMask':
            print('timeMask')
            qa_embed_with_decay = timeMask(qa_embed_reshape, tf_decay_factor, self.args.proj_len)
        elif self.args.decay_factor_in_input == 'timeJoint':
            print('timeJoint')
            qa_embed_with_decay = timeJoint(qa_embed_reshape, tf_decay_factor, self.args.proj_len)
        else:
            print('notime')
            qa_embed_with_decay = qa_embed_reshape
        qa_embed_data = tf.reshape(qa_embed_with_decay, [-1, self.args.maxlen, feature_dim])
        slice_qa_embed_data = tf.split(qa_embed_data, self.args.maxlen, 1)

        prediction = []
        reuse_flag = False
        for i in range(self.args.maxlen):
            if i!=0:
                reuse_flag = True
            
            q = tf.squeeze(slice_q_embed_data[i], 1)
            self.correlation_weight = self.memory.attention(q)

            # add decay factor in recurrent layer
            if self.args.decay_factor_in_recurrent:
                t = slice_t_data[i]
                nreps = slice_nreps_data[i]
                decay_factor = tf.concat([t, nreps], axis=1)
                self.new_value_memory  =self.memory.write_decay(self.correlation_weight, decay_factor, reuse=reuse_flag)

            self.read_content = self.memory.read(self.correlation_weight)

            qa = tf.squeeze(slice_qa_embed_data[i] ,1)
            self.new_value_memory = self.memory.write_qa(self.correlation_weight, qa, reuse=reuse_flag)

            mastery_level_prior_knowledge = tf.concat([self.read_content, q], 1)
            summary_vectot = linear(mastery_level_prior_knowledge, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
            summary_vectot = tf.tanh(summary_vectot)
            pred_logits = linear(summary_vectot, 1, name='Prediction', reuse=reuse_flag)
            prediction.append(pred_logits)
        
        self.pred_logits = tf.reshape(tf.stack(prediction, axis=1), [-1, self.args.maxlen]) 

        # loss: standard cross entropy loss
        # ignore '-1' label example
        target_1d = tf.reshape(self.tf_y, [-1])
        pred_logits_1d = tf.reshape(self.pred_logits, [-1])

        index = tf.where(tf.not_equal(target_1d, tf.constant(-1, dtype=tf.float32)))
        self.filtered_targets = tf.squeeze(tf.gather(target_1d, index), axis=1)
        filtered_logits = tf.squeeze(tf.gather(pred_logits_1d, index), axis=1)

        self.filtered_predict_prob = tf.nn.sigmoid(filtered_logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=self.filtered_targets))
    
        # optimize
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta2=0.98)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # optimizer = tf.train.MomentumOptimizer(self.lr, self.args.momentum)
        # grads, vrbs = zip(*optimizer.compute_gradients(self.loss))
        # grad, _ = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
        # self.train_op = optimizer.apply_gradients(zip(grad, vrbs), global_step=self.global_step)

        # self.tr_vrbs = tf.trainable_variables()
        # self.num_tr_vrbs = 0
        # for i in self.tr_vrbs:
        #     print(i.name, i.get_shape())
        #     tmp_num = 1
        #     for dim in i.get_shape():
        #         tmp_num *= dim.value
        #     self.num_tr_vrbs += tmp_num
        # print('Number of trainable variables: %d' % self.num_tr_vrbs)