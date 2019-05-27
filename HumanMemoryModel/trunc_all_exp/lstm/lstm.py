import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pickle as pkl
import tensorflow as tf
import numpy as np
from sklearn import metrics
import time
from utils import timeJoint, timeMask, split_data, batch_generator, pass_rate, lstm_net


dataset = 'mnemosyne'
load_path = os.path.join(os.path.dirname(os.getcwd()), 'trunc_feature/%s_trunc_lstm.pkl' % (dataset))
t0 = time.time()
with open(load_path, 'rb') as f:
    train_q, train_qa, train_decay_factor, train_targets, valid_q, valid_qa, valid_decay_factor, valid_targets = pkl.load(f)
print('read data time: ', time.time()-t0)

n_items = np.max(train_q)
train_seq_len = np.sum(train_q!=0, axis=1)
valid_seq_len = np.sum(valid_q!=0, axis=1)

# network parameters
n_hidden = 64
feature_dim = 32
proj_len = 32  # paper:30
cont_len = 32
lr = 5e-2
n_epochs = 100
batch_size = 8
# num_step = train_qa.shape[1]
n_batches = train_qa.shape[0] // batch_size
add_time_method = 'no time'   # timeJoint / timeMask / concate / no time
decay_factor_dim = train_decay_factor.shape[-1]
is_training = True

# tf placeholder
tf_qa = tf.placeholder(tf.int32, [None, None])  # qa
tf_decay_factor = tf.placeholder(tf.float32, [None, None, decay_factor_dim])    # [tlast, nreps, stu_id]
tf_actions_flatten = tf.placeholder(tf.int32, [None])     # q
tf_targets_flatten = tf.placeholder(tf.float32, [None])   # 0 / 1 / -1, -1 is ignored
tf_seq_len = tf.placeholder(tf.int32, [None])
tf_lstm_init_c = tf.placeholder(tf.float32, [None, n_hidden])
tf_lstm_init_h = tf.placeholder(tf.float32, [None, n_hidden])
tf_batch_size = tf.shape(tf_qa)[0]

# feature embedding
with tf.variable_scope('Embedding'):
    qa_embed = tf.get_variable('feature_embed', [n_items*2+1, feature_dim], \
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
tf_qa_embed = tf.nn.embedding_lookup(qa_embed, tf_qa)

# add decay factor
tf_qa_embed_reshape = tf.reshape(tf_qa_embed, [-1, feature_dim])
tf_decay_factor_reshape = tf.reshape(tf_decay_factor, [-1, decay_factor_dim])
if add_time_method == 'timeJoint':
    print('time joint')
    tf_features_with_decay = timeJoint(tf_qa_embed_reshape, tf_decay_factor_reshape, proj_len)
elif add_time_method == 'timeMask':
    print('time mask')
    tf_features_with_decay = timeMask(tf_qa_embed_reshape, tf_decay_factor_reshape, cont_len)
elif add_time_method == 'concate':
    print('time concat')
    tf_features_with_decay = tf.concat([tf_qa_embed_reshape, tf_decay_factor_reshape], axis=1)
    feature_dim += decay_factor_dim
elif add_time_method == 'no time':
    print('no time')
    tf_features_with_decay = tf_qa_embed_reshape
else:
    print('add time method wrong')
    exit()
tf_features_with_decay = tf.reshape(tf_features_with_decay, [-1, tf.shape(tf_qa)[1], feature_dim])

# lstm net
last_state, indexed_logits = lstm_net(tf_features_with_decay, tf_seq_len, tf_actions_flatten, tf_lstm_init_c, tf_lstm_init_h, n_hidden, n_items)

# optimize
index = tf.where(tf.not_equal(tf_targets_flatten, tf.constant(-1, dtype=tf.float32)))
filtered_targets = tf.squeeze(tf.gather(tf_targets_flatten, index), axis=1)
filtered_logits = tf.squeeze(tf.gather(indexed_logits, index), axis=1)
predict_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=filtered_targets,
                                            logits=filtered_logits))
filtered_predict_prob = tf.nn.sigmoid(filtered_logits)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=n_batches, decay_rate=0.99)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(predict_loss, global_step=global_step)

tf_vrbs = tf.trainable_variables()
num_tf_vrbs = 0
for v in tf_vrbs:
    print(v.name, v.get_shape())
    tmp_num = 1
    for dim in v.get_shape():
        tmp_num *= dim.value
    num_tf_vrbs += tmp_num
print('number of trainable variables: %d' % num_tf_vrbs)

# record parameters 
tag = int(time.time())
config_name = 'logs/%d.config' % tag
log_name = 'logs/%d.log' % tag
# config_name = 'logs/%s_%d_%s.config' % (dataset, n_hidden, add_time_method)
# log_name = 'logs/%s_%d_%s.log' % (dataset, n_hidden, add_time_method)
config = 'dataset: %s\n' % load_path
config += 'n_items: %d\n' % n_items
config += 'n_hidden: %d\n' % n_hidden
config += 'feature_embed_dim: %d\n' % feature_dim
config += 'method to add time: %s\n' % add_time_method
config += 'time_jointLen: %d\n' % proj_len
config += 'time_maskLen: %d\n' % cont_len
config += 'learning_rate: %f\n' % lr
config += 'batch size: %d\n' % batch_size
config += 'n_epochs: %d\n' % n_epochs
config += 'numebr of trainable  variables: %d\n' % num_tf_vrbs
with open(config_name, 'w') as f:
    f.write(config)

# data analysis
print('train/valid data shape: ', train_q.shape, valid_q.shape)
print('train set pass rate %f' % pass_rate(train_qa, n_items))
print('valid set pass rate %f' % pass_rate(valid_qa, n_items))

# begin train
train_valid_data = [train_q, train_qa, train_targets, train_decay_factor, train_seq_len, valid_q, valid_qa, valid_targets, valid_decay_factor, valid_seq_len]
batch_gen = batch_generator(train_valid_data, batch_size, shuffle=True)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(n_epochs):
        train_loss = 0
        valid_pred_list = []
        valid_target_list = []
        t0 = time.time()
        for j in range(n_batches):
            tt0 = time.time()
            batch_actions, batch_qa, batch_targets, batch_decay_factor, batch_seq_len, \
            batch_valid_actions, batch_valid_qa, batch_valid_targets, batch_valid_decay_factor, batch_valid_seq_len = batch_gen.__next__()
            # print(batch_decay_factor.shape)
            # train
            batch_targets_flatten = np.reshape(batch_targets, [-1])
            batch_actions_flatten = np.reshape(batch_actions, [-1])
            init_lstm_c = np.zeros([batch_actions.shape[0], n_hidden]).astype(np.float64)
            feed_dict = {tf_qa:batch_qa, tf_decay_factor:batch_decay_factor,
                         tf_actions_flatten:batch_actions_flatten, tf_targets_flatten:batch_targets_flatten, 
                         tf_seq_len:batch_seq_len, tf_lstm_init_c:init_lstm_c, tf_lstm_init_h:init_lstm_c}
            loss_, _ , lstm_state = sess.run([predict_loss, train_op, last_state], feed_dict=feed_dict)
            train_loss += loss_
            # tt1 = time.time()
            # print('train one batch time, ', tt1-tt0)

            # valid
            batch_valid_actions_flatten = np.reshape(batch_valid_actions, [-1])
            batch_valid_targets_flatten = np.reshape(batch_valid_targets, [-1])
            valid_feed_dict = {tf_qa:batch_valid_qa, tf_decay_factor:batch_valid_decay_factor,
                         tf_actions_flatten:batch_valid_actions_flatten, tf_targets_flatten:batch_valid_targets_flatten, 
                         tf_seq_len:batch_valid_seq_len, tf_lstm_init_c:lstm_state[0], tf_lstm_init_h:lstm_state[1]}            
            batch_valid_preds = sess.run(filtered_predict_prob, feed_dict=valid_feed_dict)
            valid_target_list.append(batch_valid_targets_flatten)
            valid_pred_list.append(batch_valid_preds)

            # print('one batch time: ', time.time()-tt0)

        train_loss /= n_batches

        all_valid_targets = np.concatenate(valid_target_list, axis=0)
        all_valid_preds = np.concatenate(valid_pred_list, axis=0)
        # print(all_valid_preds.shape, all_valid_preds.shape) # 91150,

        valid_auc = metrics.roc_auc_score(all_valid_targets, all_valid_preds)
        all_valid_preds[all_valid_preds>0.5] = 1
        all_valid_preds[all_valid_preds<=0.5] = 0
        valid_acc = metrics.accuracy_score(all_valid_targets, all_valid_preds) 
        valid_f1 = metrics.f1_score(all_valid_targets, all_valid_preds)
    
        records = 'Epoch %d/%d, train loss:%3.5f, valid auc:%f, valid acc:%3.5f, valid f1:%3.5f' % \
                        (i+1, n_epochs, train_loss, valid_auc, valid_acc, valid_f1)     
        print(records)
        with open(log_name, 'a') as f:
            f.write(records+'\n')

        print(time.time()-t0)

 
