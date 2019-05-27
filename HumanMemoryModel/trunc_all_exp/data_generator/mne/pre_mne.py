import os
import numpy as np
import pickle as pkl
import pandas as pd
import time
import math
from collections import defaultdict

max_len = 2174   # max len = 2174
last_k_items = 1
dump_path = 'mnemosyne%d_trunc.pkl' % max_len
csv_file = 'mnemosyne_history.csv'
#csv_file = 'right_wrong_mnemosyne_history.csv'
df = pd.read_csv(csv_file)


def add_colume_right_wrong():
    each_inter_success = defaultdict(int)
    history_right = []
    history_wrong = []
    for index, row in df.iterrows():
        stu_item = row['student_id']
        tmp_right = each_inter_success[stu_item]
        history_right.append(tmp_right)
        history_wrong.append(row['nreps']-1-tmp_right)
        if row['outcome']:
            each_inter_success[stu_item] += 1
    df['history_right'] = history_right
    df['history_wrong'] = history_wrong
    df.to_csv('right_wrong_'+csv_file)


def pre_process_csv():
    # fitered data based on max_len
    stu_seq = df.groupby('user_id', as_index=True)
    filtered_stu_seq = [stu for stu in stu_seq if len(stu[1])<=max_len]

    # assign each item a id, begin from 1
    all_items = set()
    for stu in filtered_stu_seq:
        tmp_items = set(stu[1]['module_id'])
        all_items = all_items | tmp_items
    items_id = {item:idx+1 for idx, item in enumerate(all_items)}

    # truncate sequence
    all_inter_data = []
    for stu in filtered_stu_seq:
        inter_per_student = []
        stu_id, stu_pd = stu[0], stu[1]
        stu_items = stu_pd.groupby('student_id')
        student_cut_loc = (stu_items.nreps.max()-last_k_items).to_dict()
        stu_list = stu_pd[['student_id', 'module_id', 'outcome', 'timestamp', 'tlast', 'nreps', 'history_right', 'history_wrong']].values.tolist()
        i = 0
        while i < len(stu_list):
            interaction = stu_list[i]
            if interaction[5] > student_cut_loc[interaction[0]]:
                inter_per_student.append(stu_list[:i+1])
                stu_list.pop(i)
            else:
                i += 1
        all_inter_data.append(inter_per_student)
    
    # feature generated
    q = []
    qa = []
    decay_factor = []
    nreps = []
    max_len1 = 0

    pass_all = 0
    fail_all = 0
    pass_valid = 0
    fail_valid = 0
   
    for i in range(len(all_inter_data)):
        inter_per_student = all_inter_data[i]
        for inter_same_student in inter_per_student:
            q_episode = []
            qa_episode = []
            decay_factor_episode = []
            nreps_episode = []
            for j in range(len(inter_same_student)):
                inter = inter_same_student[j]
                tmp_act = items_id[inter[1]]
                tmp_ans = 1 if inter[2] else 0
                tmp_decay_factor = [(inter[3]-inter[4])/(60*60*24), math.sqrt(1+inter[6]), math.sqrt(1+inter[7])]    # [tlast, history_right, history_wrong]
                q_episode.append(tmp_act)
                qa_episode.append(tmp_act+tmp_ans*len(items_id))
                decay_factor_episode.append(tmp_decay_factor)
                nreps_episode.append(inter[5])
            q.append(q_episode)
            qa.append(qa_episode)
            decay_factor.append(decay_factor_episode)
            nreps.append(nreps_episode)
            max_len1 = max(max_len1, len(q_episode))

    with open(dump_path, 'wb') as f:
        pkl.dump([q, qa, decay_factor, nreps], f)

    print('student number: %d' % len(all_inter_data))
    print('items number: %d' % len(items_id))
    print('truncated episodes number: %d' % len(q))
    print('max episode length: %d' % max_len1)

#add_colume_right_wrong()
pre_process_csv()
