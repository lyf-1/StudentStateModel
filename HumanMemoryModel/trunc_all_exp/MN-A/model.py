import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
import shutil
from memory import MN
import utils
import time


class Model:
    def __init__(self, args, sess, name='SP'):
        self.args = args
        self.name = name
        self.sess = sess
        self.model_dir = str(int(time.time()))
        # self.model_dir = '%s_%d_%d' % (self.args.dataset, self.args.key_mem_state_dim, self.args.final_fc_dim)
        print('log file: ', self.model_dir)

        self.create_model()

    def create_model(self):
        self.q_data = tf.placeholder(tf.int32, [None, self.args.seq_len], name='q_data')
        self.qa_data = tf.placeholder(tf.int32, [None, self.args.seq_len], name='qa_data')
        self.target = tf.placeholder(tf.float32, [None, self.args.seq_len], name='target')
        self.decay_factor = tf.placeholder(tf.float32, [None, self.args.seq_len, self.args.decay_dim], name='decay_factor')
        self.real_seq_len = tf.placeholder(tf.int32, [None])
        
        init_memory = tf.zeros([tf.shape(self.q_data)[0], self.args.n_items+1, self.args.mem_state_dim], dtype=tf.float32, name='init_mem_state')
        self.memory = MN(self.args.n_items+1, self.args.mem_state_dim, init_memory, name='MN')

        with tf.variable_scope('Embedding'):
            q_embed_mtx = tf.get_variable('q_embed', [self.args.n_items+1, self.args.feature_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        # qa embedding, same as EERNN, 18-AAAI
        zeros = tf.zeros([self.args.n_items+1, self.args.feature_dim], dtype=tf.float32)
        qa_embed_mtx_1 = tf.concat([zeros, q_embed_mtx], 1)
        qa_embed_mtx_2 = tf.concat([q_embed_mtx[1:], zeros[1:]], 1)
        qa_embed_mtx = tf.concat([qa_embed_mtx_1, qa_embed_mtx_2], 0)
        qa_embed_data = tf.nn.embedding_lookup(qa_embed_mtx, self.qa_data)   
        slice_qa_embed_data = tf.split(qa_embed_data, self.args.seq_len, 1)  
        
        q_embed_data = tf.nn.embedding_lookup(q_embed_mtx, self.q_data)     # [batch_size, seq_len, feature_dim]
        slice_q_data = tf.split(self.q_data, self.args.seq_len, 1)          # lsit:[a, b, ...]  a:[batch_size, 1]
        slice_decay_factor = tf.split(self.decay_factor, self.args.seq_len, 1)

        # consine similarity        
        q_embed_reshape = tf.reshape(q_embed_data, [-1, self.args.feature_dim])                         # [batch_size*seqlen, feature_dim]
        a = tf.expand_dims(tf.sqrt(tf.reduce_sum(q_embed_reshape*q_embed_reshape, 1)), axis=1)          # [batch_size*seqlen, 1]
        b = tf.transpose(tf.expand_dims(tf.sqrt(tf.reduce_sum(q_embed_mtx*q_embed_mtx, 1)), axis=1))    # [1, n_items+1]
        c = tf.matmul(q_embed_reshape, tf.transpose(q_embed_mtx))                                          # [batch_size*seqlen, n_items+1]
        score_data = tf.div(c, tf.matmul(a, b)+1e-8, name="scores")                          
        score_data = tf.reshape(score_data, [-1, self.args.seq_len, self.args.n_items+1])               # [batch_size, seqlen, n_items+1]
        slice_score = tf.split(score_data, self.args.seq_len, 1)

        prediction =[]
        reuse_flag = False
        for i in range(self.args.seq_len):
            if i!=0:
                reuse_flag = True

            q = tf.squeeze(slice_q_data[i], 1)            
            decay_factor = tf.squeeze(slice_decay_factor[i], 1)  # [batch_size, self.args.decay_dim]
            self.new_key_memory = self.memory.write_decay(q, decay_factor, reuse=reuse_flag)
            
            score = tf.squeeze(slice_score[i], 1)       # [batch_size, n_items+1] 
            self.read_content = self.memory.read(score, reuse=reuse_flag)

            qa = tf.squeeze(slice_qa_embed_data[i], 1) 
            self.new_key_memory = self.memory.write_qa(q, qa, reuse=reuse_flag)

            mastery_level_prior_knowledge = self.read_content
            summary_vector = utils.linear(mastery_level_prior_knowledge, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
            summary_vector = tf.tanh(summary_vector)
            pred_logits = utils.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)
            prediction.append(pred_logits)

        self.pred_logits = tf.reshape(tf.stack(prediction, axis=1), [-1, self.args.seq_len])
        
        idx = tf.transpose(tf.stack([tf.range(tf.shape(self.real_seq_len)[0]), self.real_seq_len-1], 0))
        self.valid_logits = tf.gather_nd(self.pred_logits, idx)  
        self.valid_target = tf.gather_nd(self.target, idx)  # [batch_size,]
        self.valid_pred = tf.sigmoid(self.valid_logits)

        idx = tf.cast(idx, dtype=tf.int64)
        tf_logits_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(values=-1-self.valid_logits, indices=idx, dense_shape=tf.shape(self.pred_logits, out_type=tf.int64)))
        tf_target_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(values=-1-self.valid_target, indices=idx, dense_shape=tf.shape(self.pred_logits, out_type=tf.int64)))
        self.train_logits = self.pred_logits + tf_logits_mask
        self.train_target = self.target + tf_target_mask    # [batch_size, args.seq_len]


        target_1d = tf.reshape(self.train_target, [-1])
        pred_logits_1d = tf.reshape(self.train_logits, [-1])
        index = tf.where(tf.not_equal(target_1d, tf.constant(-1, dtype=tf.float32)))
        filtered_targets = tf.gather(target_1d, index)
        filtered_logits = tf.gather(pred_logits_1d, index)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=filtered_targets,
                                    logits=filtered_logits))

        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, 
                                             decay_steps=self.args.anneal_interval, decay_rate=0.667, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        
        self.tr_vrbs = tf.trainable_variables()
        self.num_tr_vrbs = 0
        for i in self.tr_vrbs:
            print(i.name, i.get_shape())
            tmp_num = 1
            for dim in i.get_shape():
                tmp_num *= dim.value
            self.num_tr_vrbs += tmp_num
        print('Number of trainable variables: %d' % self.num_tr_vrbs)
        self.saver = tf.train.Saver()

    def train(self, q_data, qa_data, decay_factor_data, target_data, real_seq_len):
        shuffle_index = np.random.permutation(q_data.shape[0])
        q_data_shuffled = q_data[shuffle_index]
        qa_data_shuffled = qa_data[shuffle_index]  
        decay_factor_shuffled = decay_factor_data[shuffle_index]
        target_data_shuffled = target_data[shuffle_index]
        real_seq_len_shuffled = real_seq_len[shuffle_index]
        
        n_batches = q_data.shape[0] // self.args.batch_size + 1
        self.sess.run(tf.global_variables_initializer())

        config_path = os.path.join(self.args.log_dir, self.model_dir+'.config')
        log_path = os.path.join(self.args.log_dir, self.model_dir+'.log')
        with open(config_path, 'w') as file:
            file.write(str(self.args))
            file.write(str(self.num_tr_vrbs))

        for epoch in range(self.args.num_epochs):
            t0 = time.time()
            epoch_loss = 0
            valid_pred_list = []
            valid_target_list = []
            for steps in range(n_batches):
                tt0 = time.time()
                if steps*self.args.batch_size >= q_data_shuffled.shape[0]:
                    break
                batch_q = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                batch_qa = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                batch_decay_factor = decay_factor_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :, :]
                batch_target = target_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                batch_real_seq_len = real_seq_len_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size]

                feed_dict = {self.q_data:batch_q, self.qa_data:batch_qa, self.target:batch_target, self.decay_factor:batch_decay_factor, self.real_seq_len:batch_real_seq_len}
                loss_, _, batch_valid_target, batch_valid_pred = self.sess.run([self.loss, self.train_op, self.valid_target, self.valid_pred], feed_dict=feed_dict)
                epoch_loss += loss_
                valid_target_list.append(batch_valid_target)
                valid_pred_list.append(batch_valid_pred)
                print(time.time()-tt0)

            epoch_loss = epoch_loss / n_batches

            all_valid_target = np.concatenate(valid_target_list, axis=0)
            all_valid_pred = np.concatenate(valid_pred_list, axis=0)
            valid_auc = metrics.roc_auc_score(all_valid_target, all_valid_pred)
            all_valid_pred[all_valid_pred>0.5] = 1.
            all_valid_pred[all_valid_pred<=0.5] = 0.
            valid_acc = metrics.accuracy_score(all_valid_target, all_valid_pred) 
            valid_f1 = metrics.f1_score(all_valid_target, all_valid_pred) 

            records = 'Epoch %d/%d, train loss:%f, valid auc:%3.5f, valid acc:%3.5f, valid f1:%3.5f' % \
                     (epoch+1, self.args.num_epochs, epoch_loss, valid_auc, valid_acc, valid_f1)            
            print(records)
            with open(log_path, 'a') as f:
                f.write(records+'\n')
            print(time.time()-t0)
