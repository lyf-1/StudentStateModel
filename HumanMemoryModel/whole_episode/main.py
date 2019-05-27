import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"   
import pickle as pkl
import tensorflow as tf
import numpy as np
from sklearn import metrics
from utils import split_data, pass_rate, batch_generator
from model import DKT, DKVMN_model
import math


class Hyperparameters():
    def __init__(self):
        self.model_name = 'lstm'
        self.train_flag = True
        self.saved_path = 'models/%s.ckpt' % self.model_name
 
        self.dataset = 'duolingo'
        self.skill_num = None
        self.maxlen = None
        
        self.epochs = 100
        self.batch_size = 32
        self.lr = 0.001
        self.l2_param = .0

        # parameters only for lstm
        self.embedding_size = 32
        self.hidden_units = 50
        
        # parameters only for dkvmn
        self.mem_size = 30
        self.key_mem_state_dim = 50
        self.value_mem_state_dim = 50
        self.final_fc_dim = 50

        # add decay factor parameters
        self.proj_len = 32
        self.decay_factor_in_input = 'noTime'
        self.decay_factor_in_recurrent = False


hp = Hyperparameters()

# load data
if hp.dataset == 'duolingo':
    load_path = 'data/duolingo_20000_500_13439.pkl'
    hp.skill_num = 13439
    hp.maxlen = 499
elif hp.dataset == 'efc':
    load_path = 'data/efc_4000_500_50.pkl'
    hp.skill_num = 50
    hp.maxlen = 499
else:
    print('no dataset')
    exit()
q, qa, tlast, nreps = pkl.load(open(load_path, 'rb'))
ans = ((qa-1)//hp.skill_num).astype(np.float32)

if hp.model_name == 'lstm':
    q, qa, ans, tlast, nreps = q[:, 1:], qa[:, :-1], ans[:, 1:], tlast[:, :-1], nreps[:, :-1]
else:
    q, qa, ans, tlast, nreps = q[:, 1:], qa[:, 1:], ans[:, 1:], tlast[:, 1:], nreps[:, 1:]
datalist_ = [q, qa, ans, tlast, nreps]
train_q, valid_q, train_qa, valid_qa, train_targets, valid_targets, \
         train_tlast, valid_tlast, train_nreps, valid_nreps = split_data(datalist_, 0.8)

# print('train/valid data shape: ', train_q.shape, valid_q.shape)
# print('dataset pass rate %f' % pass_rate(qa, hp.skill_num))
# print('train set pass rate %f' % pass_rate(train_qa, hp.skill_num))
# print('valid set pass rate %f' % pass_rate(valid_qa, hp.skill_num))

# build model
if hp.model_name == 'lstm':
    model = DKT(hp)
else:
    model = DKVMN_model(hp)
model.creat_graph()

# begin train
train_batch_gen = batch_generator([train_q, train_qa, train_targets, train_tlast, train_nreps], hp.batch_size, shuffle=True)
valid_batch_gen = batch_generator([valid_q, valid_qa, valid_targets, valid_tlast, valid_nreps], hp.batch_size, shuffle=False)
train_steps = train_qa.shape[0] // hp.batch_size
valid_steps = int(math.ceil(valid_q.shape[0]/hp.batch_size))

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
best_auc = 0

if hp.train_flag:
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(hp.epochs):
            train_loss = 0
            for j in range(train_steps):
                batch_actions, batch_features, batch_targets, batch_t, batch_nreps = train_batch_gen.__next__()
                batch_real_seq_len = np.sum(batch_targets!=-1, axis=1)

                feed_dict = {model.tf_action:batch_actions, model.tf_x:batch_features, model.tf_y:batch_targets, 
                            model.tf_time:batch_t, model.tf_nreps:batch_nreps, model.tf_real_seq_len:batch_real_seq_len}
                loss_, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict)
                train_loss += loss_
            train_loss /= train_steps
    
            valid_pred_list = []
            valid_target_list = []
            valid_loss = 0
            for s in range(valid_steps):
                batch_valid_actions, batch_valid_features, batch_valid_targets, batch_valid_t, batch_valid_nreps = valid_batch_gen.__next__()
                batch_valid_real_seq_len = np.sum(batch_valid_targets!=-1, axis=1)

                valid_feed_dict = {model.tf_x:batch_valid_features, model.tf_action:batch_valid_actions, model.tf_y:batch_valid_targets, 
                                model.tf_time:batch_valid_t, model.tf_nreps:batch_valid_nreps, model.tf_real_seq_len:batch_valid_real_seq_len}
                valid_filtered_targets, valid_filtered_preds, batch_valid_loss = sess.run([model.filtered_targets, model.filtered_predict_prob, model.loss], feed_dict=valid_feed_dict)

                valid_target_list.append(valid_filtered_targets)
                valid_pred_list.append(valid_filtered_preds)
                valid_loss += batch_valid_loss
            valid_loss /= valid_steps    
            all_valid_targets = np.concatenate(valid_target_list, axis=0)
            all_valid_preds = np.concatenate(valid_pred_list, axis=0)

            valid_auc = metrics.roc_auc_score(all_valid_targets, all_valid_preds)
            all_valid_preds[all_valid_preds>0.5] = 1
            all_valid_preds[all_valid_preds<=0.5] = 0
            valid_acc = metrics.accuracy_score(all_valid_targets, all_valid_preds) 

            records = 'Epoch %d/%d, train loss:%3.5f, valid auc:%f, valid acc:%3.5f, best auc:%3.5f' % \
                            (i+1, hp.epochs, train_loss, valid_auc, valid_acc, best_auc)     
            print(records)
            with open(model.log_name, 'a') as f:
                f.write(records+'\n')

            if best_auc < valid_auc:
                best_auc = valid_auc
                # saver.save(sess, hp.saved_path)

else:
    with tf.Session(config=config) as sess: 
        saver.restore(sess, hp.saved_path)

        batch_valid_actions, batch_valid_features, batch_valid_targets, batch_valid_t, batch_valid_nreps = valid_batch_gen.__next__()
        batch_valid_real_seq_len = np.sum(batch_valid_targets!=-1, axis=1)

        valid_feed_dict = {model.tf_x:batch_valid_features, model.tf_action:batch_valid_actions, model.tf_y:batch_valid_targets, 
                           model.tf_time:batch_valid_t, model.tf_nreps:batch_valid_nreps, model.tf_real_seq_len:batch_valid_real_seq_len}
        valid_filtered_targets, valid_filtered_preds = sess.run([model.filtered_targets, model.filtered_predict_prob], feed_dict=valid_feed_dict)
