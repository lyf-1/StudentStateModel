'''
    const_delay: 1 / 0.2 --- efc / hlr
'''

import simulator
import numpy as np
import pickle as pkl


# trunc config
min_inter_num = 5
last_k_items = 1

# env config
env_flag = 'efc'
n_items = 30 if env_flag=='efc' else 50
const_delay = 1 if env_flag=='efc' else 0.2
n_episodes = 4000
episode_length = 200

episode_time_sequence = (np.arange(episode_length) + 1) * const_delay
np.random.seed(0)
item_decay_rates = np.exp(np.random.normal(np.log(0.077), 1, n_items))
print(item_decay_rates)
exit()
# item_decay_rates = np.linspace(0.04, 0.4, n_items)
env_kwargs = {'n_items': n_items, 'time_sequence': episode_time_sequence, 'const_delay': const_delay}
efc_env = simulator.EFCEnv(item_decay_rates=item_decay_rates, **env_kwargs)
hlr_env = simulator.HLREnv(**env_kwargs)

env = efc_env if env_flag=='efc' else hlr_env
dump_path = '%s_syn_trunc_%d_%d_%d.pkl' % (env_flag, n_episodes, episode_length, n_items)


q = []
qa = []
decay_factor = []
nreps = []
success_cnt = 0
fail_cnt = 0
for i in range(n_episodes):
    inter_per_episode = []
    env.reset()
    done = False
    while not done:
        action = np.random.choice(range(n_items))
        done, info = env.step(action)
        inter_per_episode.append([action, info[1], info[2], info[3]])
    success_cnt += np.sum(env.success_count_array)
    fail_cnt += np.sum(env.fail_count_array)

    # delete items whose interaction number is less than min_inter_num(5)
    del_item_idx_list = np.where(env.count_array < min_inter_num)[0]
    i = 0
    while i < len(inter_per_episode):
        one_inter = inter_per_episode[i]
        if one_inter[0] in del_item_idx_list:
            inter_per_episode.pop(i)
        else:
            i += 1

    # trunc sequence
    cut_loc = env.count_array - last_k_items
    i = 0
    inter_per_stu = []
    while i < len(inter_per_episode):
        one_inter = inter_per_episode[i]
        if one_inter[3] > cut_loc[one_inter[0]]:
            inter_per_stu.append(inter_per_episode[:i+1])
            inter_per_episode.pop(i)    
        else:
            i += 1    

    # q, qa, decay_factor(tlast, history_right, history_wrong), nreps
    for inter_seq in inter_per_stu:
        q_tmp = []
        qa_tmp = []
        decay_factor_tmp = []
        nreps_tmp = []
        for one_inter in inter_seq:
            act = one_inter[0] + 1
            q_tmp.append(act)
            qa_tmp.append(act+n_items*one_inter[1])
            decay_factor_tmp.append(one_inter[2])
            nreps_tmp.append(one_inter[3])
        q.append(q_tmp)
        qa.append(qa_tmp)
        decay_factor.append(decay_factor_tmp)
        nreps.append(nreps_tmp)

with open(dump_path, 'wb') as f:
    pkl.dump([q, qa, decay_factor, nreps], f, protocol=4)

# check if every item idx is used
use_flag = [0] * n_items
for q_tmp in q:
    for ele in q_tmp:
        use_flag[ele-1] = 1
print('item idx use flag: ', use_flag)
print('success/fail number: %d/%d/%f' % (success_cnt, fail_cnt, float(success_cnt)/(success_cnt+fail_cnt)))
print('all samples number: %d %d %d' % (len(q), len(qa), len(decay_factor)))
print()

