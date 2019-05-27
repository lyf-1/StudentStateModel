import numpy as np
import tensorflow as tf
import pickle as pkl


def linear(x, state_dim, name='linear', reuse=True):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        weight = tf.get_variable('weight', [x.get_shape()[-1], state_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('bias', [state_dim], initializer=tf.constant_initializer(0))
        weighted_sum = tf.nn.xw_plus_b(x, weight, bias)
        return weighted_sum

# for synthetic data, efc, hlr, tpprl
def feature_process(load_path):
    n_items = int(load_path.split('.')[0].split('_')[-1])
    _, actions, feedbacks, _, timeinterval = pkl.load(open(load_path, 'rb'))
    q_data = []
    qa_data = []
    for episode in range(len(actions)):
        q_episode = []
        qa_episode = []
        for seq_idx in range(len(actions[0])):
            action = actions[episode][seq_idx] + 1
            answer = feedbacks[episode][seq_idx]
            q_episode.append(action)
            qa_episode.append(action+answer*n_items)
        q_data.append(q_episode)
        qa_data.append(qa_episode)
    q_data = np.array(q_data)
    qa_data = np.array(qa_data)
    return q_data, qa_data, timeinterval

def duolingo_feature_process(load_path):
    max_len = int(load_path.split('.')[0].split('_')[-1])
    lexeme, answer, time = pkl.load(open(load_path, 'rb'))
    # print(len(lexeme), len(answer), len(time))

    lexeme_idx = {}
    items_cnt = 0
    for episode in range(len(lexeme)):
        for seq_idx in range(len(lexeme[episode])):
            tmp_lexeme = lexeme[episode][seq_idx] 
            if tmp_lexeme not in lexeme_idx:
                lexeme_idx[tmp_lexeme] = items_cnt
                items_cnt += 1                
    # print('items number: ', items_cnt, len(lexeme_idx))

    q_data = []
    qa_data = []
    timeinterval = []
    for episode in range(len(lexeme)):
        q_episode = [0] * max_len
        qa_episode = [0] * max_len
        timeinterval_episode = [0] * max_len
        for seq_idx in range(len(lexeme[episode])):
            tmp_lexeme = lexeme[episode][seq_idx] 
            act = lexeme_idx[tmp_lexeme] + 1 
            ans = 1 if float(answer[episode][seq_idx]) > 0.5 else 0 
            q_episode[seq_idx] = act
            qa_episode[seq_idx] = act + ans * items_cnt
            timeinterval_episode[seq_idx] = time[episode][seq_idx]
        q_data.append(q_episode)
        qa_data.append(qa_episode)
        timeinterval.append(timeinterval_episode)
    
    q_data = np.array(q_data)
    qa_data = np.array(qa_data)
    timeinterval = np.array(timeinterval) / (np.max(timeinterval))

    # print('q, qa, timeinterval shape: ', q_data.shape, qa_data.shape, timeinterval.shape)
    # print('time interval info: ', timeinterval.sum(), np.max(timeinterval), np.min(timeinterval))
    return q_data, qa_data, timeinterval

def duolingo_for_lstm(load_path):
    max_len = int(load_path.split('.')[0].split('_')[-1])
    lexeme, answer, _ = pkl.load(open(load_path, 'rb'))
    # print(len(lexeme), len(answer), len(time))

    lexeme_idx = {}
    items_cnt = 0
    for episode in range(len(lexeme)):
        for seq_idx in range(len(lexeme[episode])):
            tmp_lexeme = lexeme[episode][seq_idx] 
            if tmp_lexeme not in lexeme_idx:
                lexeme_idx[tmp_lexeme] = items_cnt
                items_cnt += 1                
    # print('items number: ', items_cnt, len(lexeme_idx))
    q_data = []
    qa_data = []
    targets = []
    for episode in range(len(lexeme)):
        q_episode = [0] * max_len
        qa_episode = [0] * max_len
        targets_episode = [-1] * max_len
        for seq_idx in range(len(lexeme[episode])):
            tmp_lexeme = lexeme[episode][seq_idx] 
            act = lexeme_idx[tmp_lexeme] 
            ans = 1 if float(answer[episode][seq_idx]) > 0.5 else 0 
            q_episode[seq_idx] = act
            qa_episode[seq_idx] = act + ans * items_cnt
            targets_episode[seq_idx] = ans
        q_data.append(q_episode)
        qa_data.append(qa_episode)
        targets.append(targets_episode)
    
    q_data = np.array(q_data)[:, 1:].astype(np.int32)
    qa_data = np.array(qa_data)[:, :-1].astype(np.int32)
    targets = np.array(targets)[:, 1:].astype(np.float32)
    # print('q, qa, targets shape: ', q_data.shape, qa_data.shape, targets.shape)
    return q_data, qa_data, targets

def feature_for_lstm(load_path):
    n_items = int(load_path.split('.')[0].split('_')[-1])
    max_len = int(load_path.split('.')[0].split('_')[-2])
    _, actions, feedbacks, _, _ = pkl.load(open(load_path, 'rb'))
    q_data = []
    qa_data = []
    targets = []
    for episode in range(len(actions)):
        q_episode = [0] * max_len
        qa_episode = [0] * max_len
        targets_episode = [-1] * max_len
        for seq_idx in range(len(actions[0])):
            action = actions[episode][seq_idx]
            answer = feedbacks[episode][seq_idx]
            q_episode[seq_idx] = action
            qa_episode[seq_idx] = action+answer*n_items
            targets_episode[seq_idx] = answer
        q_data.append(q_episode)
        qa_data.append(qa_episode)
        targets.append(targets_episode)

    q_data = np.array(q_data)[:, 1:].astype(np.int32)
    qa_data = np.array(qa_data)[:, :-1].astype(np.int32)
    targets = np.array(targets)[:, 1:].astype(np.float32)
    # print('q, qa, targets shape: ', q_data.shape, qa_data.shape, targets.shape)
    return q_data, qa_data, targets



# feature_process('data/efc_syn_data_4000_200_30.pkl')
# duolingo_feature_process('data/duolingo_2000_90_100.pkl')
# duolingo_for_lstm('data/duolingo_2000_90_100.pkl')
