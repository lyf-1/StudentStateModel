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
        # self.model_dir = str(int(time.time()))
        self.model_dir = '%s_%d_%d_%d_%d' % (self.args.dataset, self.args.mem_size, self.args.key_mem_state_dim, self.args.value_mem_state_dim, self.args.final_fc_dim)
       
        print('log file: ', self.model_dir)

        self.create_model()

    def create_model(self):
        self.q_data = tf.placeholder(tf.int32, [None, self.args.seq_len], name='q_data')
        self.qa_data = tf.placeholder(tf.int32, [None, self.args.seq_len], name='qa_data')
        self.target = tf.placeholder(tf.float32, [None, self.args.seq_len], name='target')
        self.t_data = tf.placeholder(tf.float32, [None, self.args.seq_len], name='t_data')
        self.nreps_data = tf.placeholder(tf.float32, [None, self.args.seq_len], name='nreps_data')

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
        slice_qa_embed_data = tf.split(qa_embed_data, self.args.seq_len, 1) 
        slice_t_data = tf.split(self.t_data, self.args.seq_len, 1)
        slice_nreps_data = tf.split(self.nreps_data, self.args.seq_len, 1)

        prediction = []
        reuse_flag = False
        for i in range(self.args.seq_len):
            if i!=0:
                reuse_flag = True
            
            q = tf.squeeze(slice_q_embed_data[i], 1)
            self.correlation_weight = self.memory.attention(q)

            t = slice_t_data[i]
            nreps = slice_nreps_data[i]
            decay_factor = tf.concat([t, nreps], axis=1)
            # decay_factor = t
            self.new_value_memory  =self.memory.write_decay(self.correlation_weight, decay_factor, reuse=reuse_flag)

            self.read_content = self.memory.read(self.correlation_weight)

            qa = tf.squeeze(slice_qa_embed_data[i] ,1)
            self.new_value_memory = self.memory.write_qa(self.correlation_weight, qa, reuse=reuse_flag)

            mastery_level_prior_knowledge = tf.concat([self.read_content, q], 1)
            # mastery_level_prior_knowledge = self.read_content
            summary_vectot = utils.linear(mastery_level_prior_knowledge, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag)
            summary_vectot = tf.tanh(summary_vectot)
            pred_logits = utils.linear(summary_vectot, 1, name='Prediction', reuse=reuse_flag)
            prediction.append(pred_logits)
        
        self.pred_logits = tf.reshape(tf.stack(prediction, axis=1), [-1, self.args.seq_len]) 

        # loss: standard cross entropy loss
        # ignore '-1' label example
        target_1d = tf.reshape(self.target, [-1])
        pred_logits_1d = tf.reshape(self.pred_logits, [-1])

        index = tf.where(tf.not_equal(target_1d, tf.constant(-1, dtype=tf.float32)))
        filtered_target = tf.gather(target_1d, index)
        filtered_logits = tf.gather(pred_logits_1d, index)
        self.pred = tf.sigmoid(self.pred_logits)

        # self.filtered_pred = tf.sigmoid(filtered_logits)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
       
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        learning_rate = tf.train.exponential_decay(self.args.initial_lr, global_step=self.global_step, 
                                            decay_steps=self.args.anneal_interval, decay_rate=0.667, staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
        # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

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
    
    def train(self, train_q_data, train_qa_data, train_t_data, train_nreps_data, train_target_data, 
                valid_q_data, valid_qa_data, valid_t_data, valid_nreps_data, valid_target_data):
        shuffle_index = np.random.permutation(train_q_data.shape[0])
        q_data_shuffled = train_q_data[shuffle_index]
        qa_data_shuffled = train_qa_data[shuffle_index]  
        t_data_shuffled = train_t_data[shuffle_index]
        target_data_shuffled = train_target_data[shuffle_index]
        nreps_data_shuffled = train_nreps_data[shuffle_index]

        training_step = train_q_data.shape[0] // self.args.batch_size + 1
        self.sess.run(tf.global_variables_initializer())

        config_path = os.path.join(self.args.log_dir, self.model_dir+'.config')
        log_path = os.path.join(self.args.log_dir, self.model_dir+'.log')
        with open(config_path, 'w') as file:
            file.write(str(self.args))
            file.write(str(self.num_tr_vrbs))

        best_auc = 0
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0
            for steps in range(training_step):
                if steps*self.args.batch_size >= q_data_shuffled.shape[0]:
                    break
                q_batch_seq = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                qa_batch_seq = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                t_batch_seq = t_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                nreps_batch_seq = nreps_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
                target_batch = target_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
    
                feed_dict = {self.q_data:q_batch_seq, self.qa_data:qa_batch_seq, self.target:target_batch, self.t_data:t_batch_seq, self.nreps_data:nreps_batch_seq, self.lr:self.args.initial_lr}
                loss_, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                epoch_loss += loss_
            epoch_loss = epoch_loss / training_step

            valid_step = valid_q_data.shape[0] // self.args.batch_size + 1
            valid_pred_list = []
            valid_target_list = []
            for s in range(valid_step):    
                if s*self.args.batch_size >= valid_q_data.shape[0]:
                    break          
                valid_q = valid_q_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                valid_qa = valid_qa_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                valid_t = valid_t_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                valid_nreps = valid_nreps_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                valid_target = valid_target_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
                
                valid_feed_dict = {self.q_data:valid_q, self.qa_data:valid_qa, self.target:valid_target, self.t_data:valid_t, self.nreps_data:valid_nreps}
                valid_pred = self.sess.run(self.pred, feed_dict=valid_feed_dict)

                valid_right_target = np.asarray(valid_target).reshape(-1, )
                valid_right_pred = np.asarray(valid_pred).reshape(-1, )
                valid_right_idx = np.flatnonzero(valid_right_target!=-1).tolist()
                valid_target_list.append(valid_right_target[valid_right_idx])
                valid_pred_list.append(valid_right_pred[valid_right_idx])

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
            
        #     if best_auc < valid_auc:
        #         self.save(epoch+1)
        #         best_auc = valid_auc
        # print(best_auc)

            
    
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