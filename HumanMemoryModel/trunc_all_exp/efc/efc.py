import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
from sklearn import metrics
import pickle as pkl
import time
from tqdm import tqdm
from utils import pass_rate, batch_generator

"""
efc_syn_data's item_difficulies:
[0.44937502 0.11488856 0.20490437 0.72393284 0.49838029 0.02897773
 0.19911725 0.06618463 0.06944858 0.11609443 0.08892995 0.32966564
 0.16481822 0.0869628  0.12002123 0.10749881 0.34305284 0.06271792
 0.10530629 0.03277643 0.00599434 0.14803134 0.18277239 0.03665832
 0.745131   0.01798323 0.08060526 0.06385543 0.35658928 0.33467643]


 [ 5.51314056e-01  1.68949217e-01  2.78781295e-01  8.48567724e-01
  6.62712097e-01  2.19625880e-35  3.35054696e-01  5.97593933e-02
  6.61677718e-02  1.32429451e-01  6.67278469e-02  4.42080498e-01
  1.91359341e-01  1.13939971e-01  1.26925483e-01  1.04961954e-01
  4.18110043e-01  5.58670983e-02  1.41369924e-01 -6.74480851e-36
  1.74509871e-35  1.88319072e-01  2.46639639e-01 -2.11143208e-35
  8.52643728e-01  3.81420452e-35  1.19837143e-01  1.02308802e-01
  4.77029234e-01  4.16942775e-01]
"""

dataset = 'duolingo'
t0 = time.time()
load_path_efc = os.path.join(os.path.dirname(os.getcwd()), 'trunc_feature/%s_trunc_efc.pkl' % (dataset))
with open(load_path_efc, 'rb') as f:
   train, valid = pkl.load(f)
print('read data time: ', time.time()-t0)

train_item_idx, train_tlast, train_nreps, train_qa = train[0], train[1], train[2], train[3]
valid_item_idx, valid_tlast, valid_nreps, valid_qa = valid[0], valid[1], valid[2], valid[3]
train_item_idx = train_item_idx.astype(np.int32)
valid_item_idx = valid_item_idx.astype(np.int32)
n_items = np.max(train_item_idx)
train_targets = ((train_qa-1)//n_items).astype(np.float32)
valid_targets = ((valid_qa-1)//n_items).astype(np.float32)

# parameters
batch_size = 512*2
lr = 0.1 # 5e-2
l2_param = 0.
maxgradnorm = 20
n_batches = train_item_idx.shape[0] // batch_size
n_epochs = 50
tag = time.time()
log_name = 'logs/%s_%d.log' % (dataset, tag)
config = 'batch_size:%d, lr:%f, l2_param:%f, maxgradnorm:%f' % (batch_size, lr, l2_param, maxgradnorm)
with open(log_name, 'a') as f:
    f.write(config+'\n')

# data analysis
print('train/valid data shape: ', train_item_idx.shape, valid_item_idx.shape)
print('items number: ', n_items)
print('train set pass rate %f' % pass_rate(train_targets))
print('valid set pass rate %f' % pass_rate(valid_targets))
print('log file: ', log_name)

# placeholder variables
tf_item_idx = tf.placeholder(tf.int32, [None])
tf_tlast = tf.placeholder(tf.float32, [None])
tf_nreps = tf.placeholder(tf.float32, [None])
tf_targets = tf.placeholder(tf.float32, [None])

# build model 
reg = tf.contrib.layers.l2_regularizer(scale=l2_param)
with tf.variable_scope('efc', reuse=tf.AUTO_REUSE, regularizer=reg):
    tf_item_difficulties = tf.get_variable('item_difficulties', [n_items], initializer=tf.truncated_normal_initializer(stddev=0.1))
item_diff = tf.gather(tf_item_difficulties, tf_item_idx-1)
# reg = tf.contrib.layers.l2_regularizer(scale=0.1)
# with tf.variable_scope('efc', reuse=tf.AUTO_REUSE, regularizer=reg):
#     tf_item_difficulties = tf.get_variable('item_difficulties', [1], initializer=tf.truncated_normal_initializer(stddev=0.1))
# item_diff = tf_item_difficulties[0]
pred_precall = tf.exp(-item_diff*(tf_tlast/tf_nreps))
pred_precall = tf.clip_by_value(pred_precall, .0001, .9999)

# optimize
precall_loss = -tf.reduce_mean(tf_targets*tf.log(pred_precall)+(1-tf_targets)*tf.log(1-pred_precall)) # cross_entropy
l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
total_loss = precall_loss + l2_loss
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=n_batches, decay_rate=0.99)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads, vrbs = zip(*optimizer.compute_gradients(total_loss))
clip_grads, _ = tf.clip_by_global_norm(grads, maxgradnorm)
train_op = optimizer.apply_gradients(zip(clip_grads, vrbs), global_step=global_step)

# train valid data generator
batch_gen_train = batch_generator([train_item_idx, train_tlast, train_nreps, train_targets], batch_size, shuffle=True)
batch_gen_valid = batch_generator([valid_item_idx, valid_tlast, valid_nreps, valid_targets], batch_size, shuffle=False)

saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(n_epochs)):
        t1 = time.time()
        train_loss = 0
        for j in range(n_batches):
            batch_item_idx, batch_tlast, batch_nreps, batch_targets = batch_gen_train.__next__()
            feed_dict = {tf_item_idx:batch_item_idx, tf_tlast:batch_tlast, tf_nreps:batch_nreps, tf_targets:batch_targets}
            batch_loss, _ = sess.run([total_loss, train_op], feed_dict=feed_dict)
            train_loss += batch_loss
        train_loss /= n_batches
        t2 = time.time()
        print(' train time: ', t2-t1)
        valid_step = valid_item_idx.shape[0] // batch_size + 1
        valid_pred_precall = []
        for s in range(valid_step):      
            valid_item_idx_, valid_tlast_, valid_nreps_, valid_targets_ = batch_gen_valid.__next__()    
            feed_dict = {tf_item_idx:valid_item_idx_, tf_tlast:valid_tlast_, tf_nreps:valid_nreps_, tf_targets:valid_targets_}
            valid_precall_, item_difficulties = sess.run([pred_precall, tf_item_difficulties], feed_dict=feed_dict)
            valid_pred_precall.append(valid_precall_)
        # print(valid_targets_)
        # print(valid_precall_)
        valid_preds = np.concatenate(valid_pred_precall, axis=0)
        valid_auc = metrics.roc_auc_score(valid_targets, valid_preds)
        valid_preds[valid_preds>=0.5] = 1
        valid_preds[valid_preds<0.5] = 0
        valid_acc = metrics.accuracy_score(valid_targets, valid_preds) 
        valid_f1 = metrics.f1_score(valid_targets, valid_preds)
        t3 = time.time()
        print(' valid time: ', t3-t2)

        records = 'Epoch %d/%d, train loss:%3.5f, valid auc:%f, valid acc:%3.5f, valid f1:%3.5f' % \
                        (i+1, n_epochs, train_loss, valid_auc, valid_acc, valid_f1)     
        print(records)
        # print(item_difficulties)
        with open(log_name, 'a') as f:
            f.write(records+'\n')
# save_model_path = 'saved_model/%s_.ckpt' % dataset
# saver.save(sess, save_model_path)





"""
# hlr model
tf_feature = tf.placeholder(tf.float32, [None, feature_dim])
tf_tlast = tf.placeholder(tf.float32, [None])
tf_targets = tf.placeholder(tf.float32, [None])
tf_halflife = tf.placeholder(tf.float32, [None])

reg = tf.contrib.layers.l2_regularizer(scale=0.1)
with tf.variable_scope('hlr', reuse=tf.AUTO_REUSE, regularizer=reg):
    tf_weights = tf.get_variable('weights', [feature_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf_bias = tf.get_variable('softmax_b', [1], initializer=tf.truncated_normal_initializer(stddev=0.1))
pred_halflife = tf.pow(2, tf.nn.xw_plus_b(tf_feature, tf_weights, tf_bias))
pred_halflife = tf.clip_by_value(pred_halflife, MIN_HALF_LIFE, MAX_HALF_LIFE)
pred_precall = tf.pow(2, -tf_tlast/pred_halflife)
pred_precall = tf.clip_by_value(pred_precall, .0001, .9999)
# optimize
# predict_loss = -tf.reduce_mean(tf_targets * tf.log(tf.clip_by_value(p_recall, 1e-10, 1.0)))
harflife_loss = tf.reduce_mean(tf.square(tf_halflife-pred_halflife))
precall_loss = tf.reduce_mean(tf.square(tf_targets-pred_precall))
l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
total_loss = precall_loss + harflife_param*harflife_loss + l2_loss
"""