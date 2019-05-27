import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm
import random
import time, math, copy

random.seed(0)

class DataGenerator_fm():
    def __init__(self, input_file, skill_num, max_len, shuffle_flag=False):
        """
        self.a: feature_idx
        self.b: feature_value
        self.c: answer
        """
        self.batch_id = 0
        self.skill_num = skill_num
        self.max_len = max_len
        self.shuffle_flag = shuffle_flag

        def feature_generator(skill_, ans_, learned_judge=3):
            cum_win = [0] * self.skill_num
            cum_fail = [0] * self.skill_num
                   
            feature_idx, feature_value, label = [], [], []
            for i in range(len(skill_)):
                s, a = skill_[i]-1, ans_[i]
                feature_idx.append([s, self.skill_num+s, self.skill_num*2+s])
                feature_value.append([1, cum_win[s], cum_fail[s]])
                label.append(a)
                if a:
                    cum_win[s] += 1
                else:
                    cum_fail[s] += 1
                
            return feature_idx, feature_value, label            

        self.a, self.b, self.c = [], [], []
        with open(input_file, 'r') as f:
            i = 0
            for line in f:
                line = eval(line)
                if type(line) == int:
                    line = [line]
                else:
                    line = list(line)
                if i % 3 == 0:
                    num = line[0]
                elif i % 3 == 1:
                    skill = line[:self.max_len]
                else:
                    ans = line[:self.max_len]

                    tmp_a, tmp_b, tmp_c = feature_generator(skill, ans)
                    self.a.extend(tmp_a)
                    self.b.extend(tmp_b)
                    self.c.extend(tmp_c)
                i += 1

        self.seq_num = len(self.a)
        if self.shuffle_flag:
            self.shuffle()
            
    def shuffle(self):
        data = list(zip(self.a, self.b, self.c))
        random.shuffle(data)
        self.a[:], self.b[:], self.c[:] = zip(*data)

    def next(self, batch_size):
        if self.batch_id >= self.seq_num:
            if self.shuffle_flag:
                self.shuffle()
            self.batch_id = 0
        batch_a = copy.deepcopy(self.a[self.batch_id:self.batch_id+batch_size])
        batch_b = copy.deepcopy(self.b[self.batch_id:self.batch_id+batch_size])
        batch_c = copy.deepcopy(self.c[self.batch_id:self.batch_id+batch_size])
        self.batch_id += batch_size

        batch_a = np.array(batch_a).astype(np.int32)
        batch_b = np.array(batch_b).astype(np.float32)
        batch_c = np.array(batch_c).astype(np.float32)
        return batch_a, batch_b, batch_c


class FM_MODEL():
    def __init__(self, hp, name='FM'):
        self.name = name
        self.hp = hp

        tag = int(time.time())
        self.config_name = 'logs/%s_%d.config' % (self.name, tag)
        self.log_name = 'logs/%s_%d.log' % (self.name, tag)
        with open(self.config_name, 'w') as f:
            for item in self.hp.__dict__.items():
                f.write(str(item)+'\n')

    def creat_graph(self):
        self.tf_x_idx = tf.placeholder(tf.int32, [None, self.hp.field_size], name='tf_x_idx')
        self.tf_x_value = tf.placeholder(tf.float32, [None, self.hp.field_size], name='tf_x_value')
        self.tf_y = tf.placeholder(tf.float32, [None], name='tf_y')
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[2], name="dropout_keep_fm")
        self.tf_batch_size = tf.shape(self.tf_x_idx)[0]
        
        with tf.variable_scope('Embedding'):
            first_order_embedding = tf.get_variable('first_order_embedding', [self.hp.feature_size, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))    
            second_order_embedding = tf.get_variable('second_order_emebedding', [self.hp.feature_size, self.hp.embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        
        x_value = tf.reshape(self.tf_x_value, [-1, self.hp.field_size, 1]) 
        self.embeddings = tf.nn.embedding_lookup(second_order_embedding, self.tf_x_idx)      # [batch_size, field_size, embedding_size]
        self.embeddings = tf.multiply(self.embeddings, x_value)

        # ---------- first order term --------------------
        self.y_first_order = tf.nn.embedding_lookup(first_order_embedding, self.tf_x_idx)                      # [batch_sizd, field_size, 1]
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, x_value), 2)
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])                # 1 --- [batch_sizd* , field_size]


        # ---------- second order term -------------------
        # sum-square-part 
        self.summed_features_emb = tf.reduce_sum(self.embeddings, axis=1)                          # [batch_size* , embedding_size]
        self.summed_features_emb_square = tf.square(self.summed_features_emb)

        # square-sum-part
        self.squared_features_emb = tf.square(self.embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, axis=1)           # [batch_size* , embedding_size]

        # second order
        self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])          # 2 --- [batch_size* , embedding_size]


        concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)  

        predict_logits = tf.layers.dense(concat_input, 1, activation=None, name='concat_ouput')       # [batch_size, 1]
        predict_logits = tf.reshape(predict_logits, [-1])

        # loss: sigmoid_cross_entropy_with_logits
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tf_y, logits=predict_logits))        

        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars]) * self.hp.l2_param   # l2_loss /= smaples_number
        self.loss = tf.add(cross_entropy, l2_loss, name='loss')
        
        # optimize
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.hp.lr, global_step=global_step, decay_steps=self.hp.decay_step, decay_rate=self.hp.decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        self.pred_prob = tf.nn.sigmoid(predict_logits)
    
    def train(self, train_dg, valid_dg):
        train_steps = int(math.ceil(train_dg.seq_num/float(self.hp.batch_size)))
        valid_steps = int(math.ceil(valid_dg.seq_num/float(self.hp.batch_size)))

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())  

            valid_preds = []
            valid_targets = []
            valid_loss = 0
            for j in range(valid_steps):
                valid_x_idx, valid_x_value, valid_y = valid_dg.next(self.hp.batch_size)

                feed_dict = {self.tf_x_idx:valid_x_idx, 
                                self.tf_x_value:valid_x_value, 
                                self.tf_y:valid_y,  
                                self.dropout_keep_fm:[1., 1.]
                                }
                valid_preds_prob, bathc_valid_loss = sess.run([self.pred_prob, self.loss], feed_dict=feed_dict)
                valid_targets.append(valid_y)
                valid_preds.append(valid_preds_prob)
                valid_loss += bathc_valid_loss

            valid_loss /= valid_steps
            valid_targets = np.concatenate(valid_targets, axis=0)
            valid_preds = np.concatenate(valid_preds, axis=0)
            # print(valid_targets.shape, np.sum(valid_targets))
            
            valid_auc = metrics.roc_auc_score(valid_targets, valid_preds)
            valid_preds[valid_preds>0.5] = 1
            valid_preds[valid_preds<=0.5] = 0
            valid_acc = metrics.accuracy_score(valid_targets, valid_preds)
            print(valid_auc, valid_acc, valid_loss)

            for i in tqdm(range(self.hp.epochs)):
                train_loss = 0
                train_preds = []
                train_targets = []
                for j in range(train_steps):
                    batch_x_idx, batch_x_value, batch_y = train_dg.next(self.hp.batch_size)

                    feed_dict = {self.tf_x_idx:batch_x_idx, 
                                 self.tf_x_value:batch_x_value, 
                                 self.tf_y:batch_y, 
                                 self.dropout_keep_fm:self.hp.dropout_keep_fm}          
                    _, batch_train_loss, batch_train_preds = sess.run([self.train_op, self.loss, self.pred_prob], feed_dict=feed_dict)
                    train_loss += batch_train_loss
                    train_preds.append(batch_train_preds)
                    train_targets.append(batch_y)

                train_loss /= train_steps
                train_targets = np.concatenate(train_targets, axis=0)
                train_preds = np.concatenate(train_preds, axis=0)
                train_auc = metrics.roc_auc_score(train_targets, train_preds)

                valid_preds = []
                valid_targets = []
                valid_loss = 0
                for j in range(valid_steps):
                    valid_x_idx, valid_x_value, valid_y = valid_dg.next(self.hp.batch_size)

                    feed_dict = {self.tf_x_idx:valid_x_idx, 
                                 self.tf_x_value:valid_x_value, 
                                 self.tf_y:valid_y,  
                                 self.dropout_keep_fm:[1., 1.]
                                 }
                    valid_preds_prob, bathc_valid_loss = sess.run([self.pred_prob, self.loss], feed_dict=feed_dict)
                    valid_targets.append(valid_y)
                    valid_preds.append(valid_preds_prob)
                    valid_loss += bathc_valid_loss

                valid_loss /= valid_steps
                valid_targets = np.concatenate(valid_targets, axis=0)
                valid_preds = np.concatenate(valid_preds, axis=0)
                # print(valid_targets.shape, np.sum(valid_targets))
                
                valid_auc = metrics.roc_auc_score(valid_targets, valid_preds)
                valid_preds[valid_preds>0.5] = 1
                valid_preds[valid_preds<=0.5] = 0
                valid_acc = metrics.accuracy_score(valid_targets, valid_preds)
                
                records = 'Epoch %d/%d, train loss:%3.5f, train auc:%3.5f, valid loss:%3.5f, valid auc:%f, valid acc:%3.5f' % \
                          (i+1, self.hp.epochs, train_loss, train_auc, valid_loss, valid_auc, valid_acc)     
                print(records)
                with open(self.log_name, 'a') as f:
                    f.write(records+'\n')


class Hyperparameters():
    def __init__(self):
        self.data_folder = 'data/dkvmn'
        self.skill_num = 100
        self.maxlen = 200
        
        self.epochs = 100
        self.batch_size = 32
        self.field_size = 3
        self.feature_size = self.skill_num * 3
        self.embedding_size = 128

        self.lr = 0.001
        self.l2_param = 0.
        self.dropout_keep_fm = [1., 1.]
        self.decay_step = 500
        self.decay_rate = 0.98
       

hp = Hyperparameters()
train_file = os.path.join(hp.data_folder, 'assist2015_train.csv')
test_file = os.path.join(hp.data_folder, 'assist2015_test.csv')

train_dg = DataGenerator_fm(train_file, hp.skill_num, hp.maxlen, shuffle_flag=True)
test_dg = DataGenerator_fm(test_file, hp.skill_num, hp.maxlen, shuffle_flag=False)  

hp.decay_step = train_dg.seq_num // hp.batch_size

model = FM_MODEL(hp)
model.creat_graph()
model.train(train_dg, test_dg)
