import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
from sklearn import metrics
from tqdm import tqdm
import math
import pickle as pkl
from data_loader import DataGenerator
from model import SELF_ATTEN_KT, DKT


class Hyperparameters():
    def __init__(self):
        self.data_folder = 'data/dkvmn'
        self.skill_num = 110
        self.maxlen = 200
        
        self.epochs = 100
        self.batch_size = 32
        self.num_blocks = 4
        self.num_heads = 1
        self.hidden_units = 64

        self.lr = 0.0007
        self.dropout = [0.5]
        self.l2_emb = 0.0
        self.l2_param = 0.0
        # self.decay_step = 500
        # self.decay_rate = 0.98


hp = Hyperparameters()
train_flag = False
saved_path = 'saved_model/selfatten_concatepos.ckpt'

train_file = os.path.join(hp.data_folder, 'assist2009_updated_train.csv')
test_file = os.path.join(hp.data_folder, 'assist2009_updated_test.csv')
train_dg = DataGenerator(train_file, hp.skill_num, hp.maxlen, shuffle_flag=True)
test_dg = DataGenerator(test_file, hp.skill_num, hp.maxlen, shuffle_flag=False)
# hp.decay_step = train_dg.seq_num // hp.batch_size


# model = DKT(hp)
model = SELF_ATTEN_KT(hp)
model.creat_graph()

train_steps = int(math.ceil(train_dg.seq_num/float(hp.batch_size)))
valid_steps = int(math.ceil(test_dg.seq_num/float(hp.batch_size)))

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
saver = tf.train.Saver()

if train_flag:
    best_auc = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())  
        for i in tqdm(range(hp.epochs)):
            train_loss = 0
            for j in range(train_steps):
                batch_x, batch_action, batch_y, batch_hist_action, batch_real_seq_len = train_dg.next(hp.batch_size)                  
                feed_dict = {model.tf_x:batch_x, 
                                model.tf_action:batch_action, 
                                model.tf_y:batch_y,
                                model.dropout_rate:hp.dropout,
                                model.tf_real_seq_len:batch_real_seq_len}
                _, batch_train_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                train_loss += batch_train_loss
            train_loss /= train_steps

            valid_preds = []
            valid_targets = []
            valid_loss = 0
            for j in range(valid_steps):
                valid_x, valid_action, valid_y, valid_hist_action, valid_real_seq_len = test_dg.next(hp.batch_size)

                feed_dict = {model.tf_x:valid_x, 
                                model.tf_action:valid_action, 
                                model.tf_y:valid_y,
                                model.dropout_rate:[0.0],
                                model.tf_real_seq_len:valid_real_seq_len}
                valid_filtered_targets, valid_filtered_preds, bathc_valid_loss = sess.run([model.filtered_targets, model.filtered_predict_prob, model.loss], 
                                                            feed_dict=feed_dict)
                valid_targets.append(valid_filtered_targets)
                valid_preds.append(valid_filtered_preds)
                valid_loss += bathc_valid_loss
                               
            valid_loss /= valid_steps
            valid_targets = np.concatenate(valid_targets, axis=0)
            valid_preds = np.concatenate(valid_preds, axis=0)
            
            valid_auc = metrics.roc_auc_score(valid_targets, valid_preds)
            valid_preds[valid_preds>0.5] = 1
            valid_preds[valid_preds<=0.5] = 0
            valid_acc = metrics.accuracy_score(valid_targets, valid_preds)
            
            if best_auc < valid_auc:
                best_auc = valid_auc
                saver.save(sess, saved_path)

            records = 'Epoch %d/%d, train loss:%3.5f, valid loss:%3.5f, valid auc:%f, valid acc:%3.5f, best_auc:%3.5f' % \
                        (i+1, hp.epochs, train_loss, valid_loss, valid_auc, valid_acc, best_auc)     
            print(records)
            with open(model.log_name, 'a') as f:
                f.write(records+'\n')

else:
    with tf.Session(config=config) as sess: 
        saver.restore(sess, saved_path)

        valid_x, valid_action, valid_y, valid_hist_action, valid_real_seq_len = test_dg.next(1)
        feed_dict = {model.tf_x:valid_x, 
                        model.tf_action:valid_action, 
                        model.tf_y:valid_y,
                        model.dropout_rate:[0.0],
                        model.tf_real_seq_len:valid_real_seq_len}
        # a = sess.run(model.weights, feed_dict=feed_dict)
        a = sess.run(model.all_preds, feed_dict=feed_dict)
        print(a.shape)

        with open('selfatten.pkl', 'wb') as f:
            pkl.dump(a, f, protocol=4)

        exit()



        # a = a[0]
        # print(valid_x.shape)
        # print(valid_x)
        # with open('weights.txt', 'w') as f:
        #     for m in range(a.shape[0]):
        #         for n in range(a.shape[1]):
        #             f.write("%.3f\t" % a[m][n])
        #         f.write('\n')
        # with open('qa.txt', 'w') as f:
        #     for ele in valid_x[0]:
        #         f.write(str(ele)+'\n')