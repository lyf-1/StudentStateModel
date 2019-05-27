import os
import numpy as np
import pickle as pkl
import pandas as pd
import math
import time


max_len = 2000   # max len = 18430, min len = 5
min_inter_num = 5
last_k_items = 1
outcome_thr = 0.859 # come from the paper
dump_path = 'duolingo%d_trunc.pkl' % max_len
csv_file = 'learning_traces.13m.csv'


def filter_inter():
    start_time = time.time()
    df = pd.read_csv(csv_file)
    # remove user-item pair whose number < min_inter_num
    df['user_lexeme'] = df['user_id']+'-'+df['lexeme_id']
    users = df['user_lexeme'].value_counts().reset_index()
    users = users['index'][users['user_lexeme']>=min_inter_num]
    filtered_df = df[df['user_lexeme'].isin(users)]
    filtered_df.sort_values('timestamp', inplace=True)
    filtered_df.to_csv('filtered_'+csv_file)
    print('original interactions number: %d' % len(df))
    print('filtered interactions number(< %d): %d' % (min_inter_num, len(filtered_df)))
    print('filter interactions whose number<%d time: %d' %(min_inter_num, time.time()-start_time))


def pre_process_csv():
    start_time = time.time()
    filtered_df = pd.read_csv('filtered_'+csv_file)
    
    # fitered data based on max_len
    stu_seq = filtered_df.groupby('user_id', as_index=True)
    filtered_stu_seq = [stu for stu in stu_seq if len(stu[1])<=max_len]
    print('student number(non_filtered/filtered_based_maxlen): %d/%d' % (len(stu_seq), len(filtered_stu_seq)))

    # assign each item a id, begin from 1
    all_items = set()
    for stu in filtered_stu_seq:
        tmp_items = set(stu[1]['lexeme_id'])
        all_items = all_items | tmp_items
    items_id = {item:idx+1 for idx, item in enumerate(all_items)}

    # truncate sequence
    all_inter_data = []
    for stu in filtered_stu_seq:
        inter_per_student = []
        stu_id, stu_pd = stu[0], stu[1]
        stu_items = stu_pd.groupby('user_lexeme')
        student_cut_loc = (stu_items.history_seen.max()-last_k_items).to_dict()
        stu_list = stu_pd[['user_lexeme', 'lexeme_id', 'p_recall', 'timestamp', 'delta', 'history_seen', 'history_correct']].values.tolist()
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
                tmp_ans = 1 if inter[2]>outcome_thr else 0
                tmp_decay_factor = [inter[4]/(60*60*24), math.sqrt(1+inter[6]), math.sqrt(1+inter[5]-inter[6])]    # [tlast, right, wrong]
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
    print('pre-process time: %d' % (time.time()-start_time))

#filter_inter()
pre_process_csv()
