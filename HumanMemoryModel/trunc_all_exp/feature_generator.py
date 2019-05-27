import os
import numpy as np
import pickle as pkl
import time


dataset = 'efc'
if dataset == 'efc':
    dump_path = 'trunc_data/efc_syn_trunc_4000_200_30.pkl'
elif dataset == 'hlr':
    dump_path = 'trunc_data/hlr_syn_trunc_4000_200_50.pkl'
elif dataset == 'mnemosyne':
    dump_path = 'trunc_data/mnemosyne2174_trunc.pkl'
elif dataset == 'duolingo':
    dump_path = 'trunc_data/duolingo2000_trunc.pkl'
else:
    print('No this dataset !')
    exit()

t0 = time.time()
with open(dump_path, 'rb') as f:
    q, qa, decay_factor, nreps = pkl.load(f)
max_len = 0
for ele in q:
    max_len = max(max_len, len(ele))
print('read data time:', time.time()-t0)

print(decay_factor[0])
print(nreps[0])



def pass_rate(qa, n_items):
    ans = (qa-1) // n_items
    num_pass = np.sum(ans==1)
    num_fail = np.sum(ans==0)
    print('num pass / num fail / num all', num_pass, num_fail, ans.shape)
    return float(num_pass) / (num_fail+num_pass)


def feature_for_efc():
    t1 = time.time()
    train, valid = [[], [], [], []], [[], [], [], []]
    for i in range(len(q)):
        assert len(q[i])==len(qa[i]) and len(q[i])==len(decay_factor[i]) and len(q[i])==len(nreps[i])
        q_episode = q[i]
        qa_episode = qa[i]
        nreps_episode = nreps[i]
        decay_factor_episode = decay_factor[i]
        for j in range(len(q_episode)-1):
            train[0].append(q_episode[j])
            train[1].append(decay_factor_episode[j][0])
            train[2].append(nreps_episode[j])
            train[3].append(qa_episode[j])
        valid[0].append(q_episode[-1])
        valid[1].append(decay_factor_episode[-1][0])
        valid[2].append(nreps_episode[-1])
        valid[3].append(qa_episode[-1])
    train = np.array(train).astype(np.float32)
    valid = np.array(valid).astype(np.float32)
      
    t2 = time.time()
    n_items = np.max(train[0])
    print('process data time: ', t2-t1)
    print('data shape: ', train.shape, valid.shape)
    print('items number: ', n_items)
    print('train set pass rate %f' % pass_rate(train[3], n_items))
    print('valid set pass rate %f' % pass_rate(valid[3], n_items))

    dump_path_efc = 'trunc_feature/%s_trunc_efc.pkl' % dataset
    with open(dump_path_efc, 'wb') as f:
        pkl.dump([train, valid], f, protocol=4)

def feature_for_lstm():
    t1 = time.time()
    train_q = np.zeros([len(q), max_len-2]).astype(np.int32)
    train_targets = np.zeros([len(q), max_len-2]).astype(np.int32)
    train_qa = np.zeros([len(q), max_len-2]).astype(np.int32)
    train_decay_factor = np.zeros([len(q), max_len-2, 3]).astype(np.float32)
    valid_q = np.zeros([len(q), 1]).astype(np.int32)
    valid_targets = np.zeros([len(q), 1]).astype(np.int32)
    valid_qa = np.zeros([len(q), 1]).astype(np.int32)
    valid_decay_factor = np.zeros([len(q), 1, 3]).astype(np.float32)
    
    for i in range(len(q)):
        length = len(q[i])
        train_qa[i, :length-2] = qa[i][:-2]
        train_decay_factor[i, :length-2, :] = decay_factor[i][:-2]
        train_q[i, :length-2] = q[i][1:-1]
        train_targets[i, :length-2] = qa[i][1:-1]
        valid_qa[i, :] = [qa[i][-2]]
        valid_decay_factor[i, :, :] = [decay_factor[i][-2]]
        valid_q[i, :] = [q[i][-1]]
        valid_targets[i, :] = [qa[i][-1]]
    
    n_items = np.max(train_q)
    print('items number: %d' % n_items)
    train_targets = ((train_targets-1)//n_items).astype(np.float32)
    valid_targets = ((valid_targets-1)//n_items).astype(np.float32)

    t2 = time.time()
    print('process data time: ', t2-t1)
    print('data shape: ')
    print(train_q.shape, train_qa.shape, train_decay_factor.shape, train_targets.shape)
    print(valid_q.shape, valid_qa.shape, valid_decay_factor.shape, valid_targets.shape)

    dump_path_lstm = 'trunc_feature/%s_trunc_lstm.pkl' % (dataset)
    with open(dump_path_lstm, 'wb') as f:
        pkl.dump([train_q, train_qa, train_decay_factor, train_targets, valid_q, valid_qa, valid_decay_factor, valid_targets], f, protocol=4)

        
def feature_for_tlstm():
    t1 = time.time()
    train_valid_q = np.zeros([len(q), max_len-1]).astype(np.int32)
    train_valid_qa = np.zeros([len(q), max_len-1]).astype(np.int32)
    train_valid_decay_factor = np.zeros([len(q), max_len-1, 3]).astype(np.float32)
    train_valid_targets = np.zeros([len(q), max_len-1]).astype(np.float32)
    for i in range(len(q)):
        assert len(q[i])==len(qa[i]) and len(q[i])==len(decay_factor[i])
        length = len(q[i]) - 1
        train_valid_qa[i, :length] = qa[i][:-1]
        train_valid_decay_factor[i, :length, :] = decay_factor[i][:-1]
        train_valid_q[i, :length] = q[i][1:]
        train_valid_targets[i, :length] = qa[i][1:]
      
    t2 = time.time()
    n_items = np.max(train_valid_q)
    train_valid_targets = ((train_valid_targets-1)//n_items).astype(np.float32)
    print('process data time: ', t2-t1)
    print('data shape: ')
    print(train_valid_q.shape, train_valid_qa.shape, train_valid_decay_factor.shape)
    print('items number: ', n_items)
    print('train set pass rate %f' % pass_rate(train_valid_qa, n_items))

    dump_path_tlstm = 'trunc_feature/%s_trunc_tlstm.pkl' % dataset
    with open(dump_path_tlstm, 'wb') as f:
        pkl.dump([train_valid_q, train_valid_qa, train_valid_decay_factor, train_valid_targets], f, protocol=4)


def feature_for_DKVMN():
    t1 = time.time()
    train_valid_q = np.zeros([len(q), max_len]).astype(np.int32)
    train_valid_qa = np.zeros([len(q), max_len]).astype(np.int32)
    train_valid_decay_factor = np.zeros([len(q), max_len, 4]).astype(np.float32)
    for i in range(len(q)):
        assert len(q[i])==len(qa[i]) and len(q[i])==len(decay_factor[i])
        length = len(q[i])
        train_valid_qa[i, :length] = qa[i]
        train_valid_decay_factor[i, :length, :3] = decay_factor[i]
        train_valid_decay_factor[i, :length, 3] = nreps[i]
        train_valid_q[i, :length] = q[i]
      
    t2 = time.time()
    n_items = np.max(train_valid_q)
    print('process data time: ', t2-t1)
    print('data shape: ')
    print(train_valid_q.shape, train_valid_qa.shape, train_valid_decay_factor.shape)
    print('items number: ', n_items)
    print('train set pass rate %f' % pass_rate(train_valid_qa, n_items))

    dump_path_DKVMN = 'trunc_feature/%s_trunc_DKVMN.pkl' % dataset
    with open(dump_path_DKVMN, 'wb') as f:
        pkl.dump([train_valid_q, train_valid_qa, train_valid_decay_factor], f, protocol=4)


# feature_for_efc()
feature_for_lstm()
# feature_for_tlstm()
# feature_for_DKVMN()

