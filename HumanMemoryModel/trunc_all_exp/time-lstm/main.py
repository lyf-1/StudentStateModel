import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
from sklearn import metrics
import pickle as pkl
import time
from tqdm import tqdm
from tlstm import TimeLstmCell, TimeLstmCell3


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]


def batch_generator(data, batch_size_, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size_ >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size_
        end = start + batch_size_
        batch_count += 1
        yield [d[start:end] for d in data]


dataset = 'mnemosyne'
t0 = time.time()
load_path = os.path.join(os.path.dirname(os.getcwd()), 'trunc_feature/%s_trunc_tlstm.pkl' % (dataset))
with open(load_path, 'rb') as f:
   q, qa, decay_factor, targets = pkl.load(f)
print('read data time: ', time.time()-t0)

# data attribute
n_items = np.max(q)    
seq_len = q.shape[1]
decay_dim = decay_factor.shape[-1]
real_seq_len = np.sum(q!=0, axis=1)
real_seq_len_check = np.sum(qa!=0, axis=1)	
assert real_seq_len.shape[0] == real_seq_len_check.shape[0] 
assert np.sum(real_seq_len!=real_seq_len_check) == 0	

# network parameters
n_hidden = 128
feature_dim = 32
batch_size = 8
lr = 5e-2
l2_param = 0.
n_batches = q.shape[0] // batch_size
n_epochs = 50

tag = time.time()
config_name = 'logs/%s_%d.config' % (dataset, tag)
log_name = 'logs/%s_%d.log' % (dataset, tag)
config = 'dataset: %s\n' % dataset
config += 'n_hidden: %d\n' % n_hidden
config += 'feature_embed_dim: %d\n' % feature_dim
config += 'learning_rate: %f\n' % lr
config += 'batch size: %d\n' % batch_size
config += 'n_epochs: %d\n' % n_epochs
config += 'n_items: %d\n' % n_items
with open(config_name, 'w') as f:
    f.write(config)
 

# some tf variables
tf_q_flatten = tf.placeholder(tf.int32, [None], name='q_data')
tf_qa = tf.placeholder(tf.int32, [None, seq_len], name='qa_data')
tf_targets = tf.placeholder(tf.float32, [None, seq_len], name='targets')
tf_decay_factor = tf.placeholder(tf.float32, [None, seq_len, decay_dim], name='decay_factor')
tf_real_seq_len = tf.placeholder(tf.int32, [None], name='real_seq_len')
tf_batch_size = tf.shape(tf_qa)[0]

# feature embedding
with tf.variable_scope('qa_Embedding'):
    qa_embed_mtx = tf.get_variable('qa_embed', [n_items*2+1, feature_dim], \
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
tf_qa_embed = tf.nn.embedding_lookup(qa_embed_mtx, tf_qa)

# lstm 
with tf.variable_scope('TimeLstmNet', reuse=tf.AUTO_REUSE):
    # lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    lstm = TimeLstmCell(n_hidden, state_is_tuple=True)
    # lstm = TimeLstmCell3(n_hidden, state_is_tuple=True)
    init_state = lstm.zero_state(tf_batch_size, tf.float32)
    state = init_state
    outputs = []
    for time_step in range(seq_len):
        lstm_output, state = lstm([tf_qa_embed[:, time_step, :], tf_decay_factor[:, time_step, :]], state)
        outputs.append(lstm_output)   # lstm_output: [batch_size, n_hidden]
    output = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden])
    softmax_w = tf.get_variable('softmax_w', [n_hidden, n_items+1], initializer=tf.truncated_normal_initializer(stddev=0.1))
    softmax_b = tf.get_variable('softmax_b', [n_items+1],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)     
    gather_index = tf.transpose(tf.stack([tf.range(tf.shape(logits)[0]), tf_q_flatten], 0))
    indexed_logits = tf.gather_nd(logits, gather_index)

pred_logits = tf.reshape(indexed_logits, [-1, seq_len])   # [batchsize, seq_len]
    
idx = tf.transpose(tf.stack([tf.range(tf.shape(tf_real_seq_len)[0]), tf_real_seq_len-1], 0))
valid_logits = tf.gather_nd(pred_logits, idx)  
valid_targets = tf.gather_nd(tf_targets, idx)  # [batch_size,]
valid_preds = tf.sigmoid(valid_logits)

idx = tf.cast(idx, dtype=tf.int64)
tf_logits_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(values=-1-valid_logits, indices=idx, dense_shape=tf.shape(pred_logits, out_type=tf.int64)))
tf_targets_mask = tf.sparse_tensor_to_dense(tf.SparseTensor(values=-1-valid_targets, indices=idx, dense_shape=tf.shape(pred_logits, out_type=tf.int64)))
train_logits = pred_logits + tf_logits_mask
train_targets = tf_targets + tf_targets_mask    # [batch_size, seq_len]

# optimize
# ignore '-1' label example
targets_1d = tf.reshape(train_targets, [-1])
pred_logits_1d = tf.reshape(train_logits, [-1])
index = tf.where(tf.not_equal(targets_1d, tf.constant(-1, dtype=tf.float32)))
filtered_targets = tf.gather(targets_1d, index)
filtered_logits = tf.gather(pred_logits_1d, index)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_targets))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=n_batches, decay_rate=0.667, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# number of trainable variables
tf_vrbs = tf.trainable_variables()
num_tf_vrbs = 0
for v in tf_vrbs:
    print(v.name, v.get_shape())
    tmp_num = 1
    for dim in v.get_shape():
        tmp_num *= dim.value
    num_tf_vrbs += tmp_num
print('number of trainable variables: %d' % num_tf_vrbs)


# data generator
batch_gen = batch_generator([q, qa, decay_factor, targets, real_seq_len], batch_size, shuffle=True)

# begin train
saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(n_epochs)):
        t1 = time.time()
        train_loss = 0
        valid_targets_list = []
        valid_preds_list = []
        for j in range(n_batches):
            batch_q, batch_qa, batch_decay_factor, batch_targets, batch_seq_len = batch_gen.__next__()
            batch_q_flatten = np.reshape(batch_q, [-1])
            feed_dict = {tf_q_flatten:batch_q_flatten, tf_qa:batch_qa, tf_decay_factor:batch_decay_factor,
                         tf_targets:batch_targets, tf_real_seq_len:batch_seq_len}

            loss_, _, batch_valid_targets, batch_valid_preds = sess.run([loss, train_op, valid_targets, valid_preds], feed_dict=feed_dict)
            train_loss += loss_
            valid_targets_list.append(batch_valid_targets)
            valid_preds_list.append(batch_valid_preds)
        t2 = time.time()
        print(' train time: ', t2-t1)
        train_loss = train_loss / n_batches

        all_valid_targets = np.concatenate(valid_targets_list, axis=0)
        all_valid_preds = np.concatenate(valid_preds_list, axis=0)
        valid_auc = metrics.roc_auc_score(all_valid_targets, all_valid_preds)
        all_valid_preds[all_valid_preds>0.5] = 1.
        all_valid_preds[all_valid_preds<=0.5] = 0.
        valid_acc = metrics.accuracy_score(all_valid_targets, all_valid_preds) 
        valid_f1 = metrics.f1_score(all_valid_targets, all_valid_preds) 
        
        t3 = time.time()
        print(' valid time: ', t3-t2)
        records = 'Epoch %d/%d, train loss:%3.5f, valid auc:%f, valid acc:%3.5f, valid f1:%3.5f' % \
                        (i+1, n_epochs, train_loss, valid_auc, valid_acc, valid_f1)              
        print(records)
        with open(log_name, 'a') as f:
            f.write(records+'\n')
        print(time.time()-t0)

save_model_path = 'saved_model/%s_.ckpt' % dataset
saver.save(sess, save_model_path)
