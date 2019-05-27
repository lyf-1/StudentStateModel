import os
import numpy as np
import tensorflow as tf
from sklearn import metrics
import shutil
from memory import DKVMN
import utils
import time

"""
    args: batch_size seq_len n_items dataset
          mem_size key_mem_state_dim value_mem_state_dim final_fc_dim
          momentum maxgradnorm 
          init_from num_epochs initial_lr anneal_interval
"""
class Model:
    def __init__(self, args, sess, name='KT'):
        self.args = args
        self.name = name
        self.sess = sess
        self.model_dir = str(int(time.time()))
        # self.model_dir = '%s_%d_%d_%d_%d' % (self.args.dataset, self.args.mem_size, self.args.key_mem_state_dim, self.args.value_mem_state_dim, self.args.final_fc_dim)
       
        print('log file: ', self.model_dir)

        self.create_model()

    def create_model(self):
        self.q_data = tf.placeholder(tf.int32, [None, self.args.seq_len], name='q_data')
        self.qa_data = tf.placeholder(tf.int32, [None, self.args.seq_len], name='qa_data')
        self.target = tf.placeholder(tf.float32, [None, self.args.seq_len], name='target')
        self.decay_factor = tf.placeholder(tf.float32, [None, self.args.seq_len, self.args.decay_dim], name='decay_factor')
        self.real_seq_len = tf.placeholder(tf.int32, [None])

        with tf.variable_scope('Memory'):
            init_key_memory = tf.get_variable('key', [self.args.mem_size, self.args.key_mem_state_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            init_value_memory = tf.get_variable('value', [self.args.mem_size, self.args.value_mem_state_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        init_value_memory = tf.tile(tf.expand_dims(init_value_memory, axis=0), tf.stack([tf.shape(self.q_data)[0], 1, 1]))

        self.memory = DKVMN(self.args.mem_size, self.args.key_mem_state_dim, self.args.value_mem_state_dim, 
                            init_key_memory, init_value_memory, name='DKVMN')
        
        with tf.variable_scope('Embedding'):
            q_embed_mtx = tf.get_variable('q_embed', [self.args.n_items+1, self.args.key_mem_state_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
            qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_items+1, self.args.value_mem_state_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        q_embed_data = tf.nn.embedding_lookup(q_embed_mtx, self.q_data)        # [batch_size, seq_len, key_mem_state_dim]
        slice_q_embed_data = tf.split(q_embed_data, self.args.seq_len, 1)  
        
        qa_embed_data = tf.nn.embedding_lookup(qa_embed_mtx, self.qa_data)     # [batch_size, seq_len, value_mem_state_dim]
        qa_embed_reshape = tf.reshape(qa_embed_data, [-1, self.args.value_mem_state_dim])

        ## add decay_factor = [time, nreps, success, fail] in input layer
        tf_decay_factor = tf.reshape(self.decay_factor, [-1, self.args.decay_dim])
        feature_dim = self.args.value_mem_state_dim
        if self.args.decay_factor_in_input == 'timeConcate':
            print('timeConcate')
            qa_embed_with_decay = tf.concat([qa_embed_reshape, tf_decay_factor], axis=1)
            feature_dim += self.args.decay_dim
        elif self.args.decay_factor_in_input == 'timeMask':
            print('timeMask')
            qa_embed_with_decay = utils.timeMask(qa_embed_reshape, tf_decay_factor, self.args.proj_len)
        elif self.args.decay_factor_in_input == 'timeJoint':
            print('timeJoint')
            qa_embed_with_decay = utils.timeJoint(qa_embed_reshape, tf_decay_factor, self.args.proj_len)
        else:
            print('notime')
            qa_embed_with_decay = qa_embed_reshape
        qa_embed_data = tf.reshape(qa_embed_with_decay, [-1, self.args.seq_len, feature_dim])
        slice_qa_embed_data = tf.split(qa_embed_data, self.args.seq_len, 1)
        slice_decay_factor = tf.split(self.decay_factor, self.args.seq_len, 1)

        prediction = []
        reuse_flag = False
        for i in range(self.args.seq_len):
            if i!=0:
                reuse_flag = True
            # step1: get correlation weight, item---concept
            q = tf.squeeze(slice_q_embed_data[i], 1)
            self.correlation_weight = self.memory.attention(q)
            # step2: memory decay based on tlast, nreps ...
            if self.args.decay_factor_in_recurrent:
                # print('time in recurrent')
                decay_factor = tf.squeeze(slice_decay_factor[i], 1)  # [batch_size, self.args.decay_dim]
                self.new_value_memory  =self.memory.write_decay(self.correlation_weight, decay_factor, reuse=reuse_flag)
            # step3: read from memory
            self.read_content = self.memory.read(self.correlation_weight)
            # step4: update memory, based on (q,a)
            qa = tf.squeeze(slice_qa_embed_data[i] ,1)
            self.new_value_memory = self.memory.write_qa(self.correlation_weight, qa, reuse=reuse_flag)

            mastery_level_prior_knowledge = tf.concat([self.read_content, q], 1)
            # mastery_level_prior_knowledge = self.read_content
            summary_vectot = utils.linear(mastery_level_prior_knowledge, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
            summary_vectot = tf.tanh(summary_vectot)
            pred_logits = utils.linear(summary_vectot, 1, name='Prediction', reuse=reuse_flag)
            prediction.append(pred_logits)
        
        self.pred_logits = tf.reshape(tf.stack(prediction, axis=1), [-1, self.args.seq_len])   # [batchsize, seq_len]
       
        idx = tf.transpose(tf.stack([tf.range(tf.shape(self.real_seq_len)[0]), self.real_seq_len-1], 0))
        self.valid_logits = tf.gather_nd(self.pred_logits, idx)  
        self.valid_target = tf.gather_nd(self.target, idx)  # [batch_size,]
        self.valid_pred = tf.sigmoid(self.valid_logits)

        idx = tf.cast(idx, dtype=tf.int64)
        tf_logits_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(values=-1-self.valid_logits, indices=idx, dense_shape=tf.shape(self.pred_logits, out_type=tf.int64)))
        tf_target_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(values=-1-self.valid_target, indices=idx, dense_shape=tf.shape(self.pred_logits, out_type=tf.int64)))
        self.train_logits = self.pred_logits + tf_logits_mask
        self.train_target = self.target + tf_target_mask    # [batch_size, args.seq_len]

        # loss: standard cross entropy loss
        # ignore '-1' label example
        target_1d = tf.reshape(self.train_target, [-1])
        pred_logits_1d = tf.reshape(self.train_logits, [-1])
        index = tf.where(tf.not_equal(target_1d, tf.constant(-1, dtype=tf.float32)))
        filtered_target = tf.gather(target_1d, index)
        filtered_logits = tf.gather(pred_logits_1d, index)
        # self.filtered_pred = tf.sigmoid(filtered_logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
       
        self.global_step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, 
                                            decay_steps=self.args.anneal_interval, decay_rate=0.667, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
        # self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        # optimizer = tf.train.MomentumOptimizer(self.lr, self.args.momentum)
        # grads, vrbs = zip(*optimizer.compute_gradients(self.loss))
        # grad, _ = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
        # self.train_op = optimizer.apply_gradients(zip(grad, vrbs), global_step=self.global_step)

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

        q_data_shuffled = q_data
        qa_data_shuffled = qa_data
        decay_factor_shuffled = decay_factor_data
        target_data_shuffled = target_data
        real_seq_len_shuffled = real_seq_len
        
        n_batches = q_data.shape[0] // self.args.batch_size + 1
        self.sess.run(tf.global_variables_initializer())

        config_path = os.path.join(self.args.log_dir, self.model_dir+'.config')
        log_path = os.path.join(self.args.log_dir, self.model_dir+'.log')
        with open(config_path, 'w') as file:
            file.write(str(self.args))
            file.write(str(self.num_tr_vrbs))

        best_auc = 0
        for epoch in range(self.args.num_epochs):
            t0 = time.time()
            epoch_loss = 0
            valid_pred_list = []
            valid_target_list = []
            for steps in range(n_batches):
                # t11 = time.time()
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
                # print(' one batch time: ', time.time()-t11)
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
            if best_auc < valid_auc:
                # self.save(epoch+1)
                best_auc = valid_auc
        print(best_auc)

            
    def test(self, test_q, test_qa):
        steps = test_q.shape[0] // self.args.batch_size
        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        pred_list = []
        target_list = []

        for s in range(steps):
            test_q_batch = test_q[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            test_qa_batch = test_qa[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
            target = test_qa_batch[:,:]
            target = target.astype(np.int)
            target_batch = (target - 1) // self.args.n_questions  
            target_batch = target_batch.astype(np.float)
            feed_dict = {self.q_data:test_q_batch, self.qa_data:test_qa_batch, self.target:target_batch}
            loss_, pred_ = self.sess.run([self.loss, self.pred], feed_dict=feed_dict)
            # Get right answer index
            # Make [batch size * seq_len, 1]
            right_target = np.asarray(target_batch).reshape(-1,1)
            right_pred = np.asarray(pred_).reshape(-1,1)
            # np.flatnonzero returns indices which is nonzero, convert it list 
            right_index = np.flatnonzero(right_target != -1.).tolist()
            # Number of 'training_step' elements list with [batch size * seq_len, ]
            pred_list.append(right_pred[right_index])
            target_list.append(right_target[right_index])

        all_pred = np.concatenate(pred_list, axis=0)
        all_target = np.concatenate(target_list, axis=0)
        # Compute metrics
        self.test_auc = metrics.roc_auc_score(all_target, all_pred)
        # Extract elements with boolean index
        # Make '1' for elements higher than 0.5
        # Make '0' for elements lower than 0.5
        all_pred[all_pred > 0.5] = 1
        all_pred[all_pred <= 0.5] = 0
        self.test_accuracy = metrics.accuracy_score(all_target, all_pred)
        print('Test auc : %3.4f, Test accuracy : %3.4f' % (self.test_auc, self.test_accuracy))

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.join(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt_name)
            return True
        else:
            return False  

    def save(self, global_step):
        model_name = 'DKVMN'
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)

    def get_w(self, q):
        self.sess.run(tf.global_variables_initializer())
        if self.load():
            print('CKPT Loaded')
        else:
            raise Exception('CKPT need')

        tf_q = tf.placeholder(tf.int32, [None])
        with tf.variable_scope('Embedding', reuse=True):
            q_embed_mtx = tf.get_variable('q_embed', [self.args.n_items+1, self.args.key_mem_state_dim],
                                           initializer=tf.truncated_normal_initializer(stddev=0.1))
        q_embed = tf.nn.embedding_lookup(q_embed_mtx, tf_q)
        correlation_weight = self.memory.attention(q_embed)

        weight = self.sess.run(correlation_weight, feed_dict={tf_q: q})
        return weight