import os
import pandas as pd
import numpy as np
import math
import random
import time


random.seed(0)

data_folder = 'data/assist09'
csv_file = 'skill_builder_data_corrected.csv'
min_len = 2
max_len = 200

def pre_assisment():
    dataset_path = os.path.join(data_folder, csv_file)
    df = pd.read_csv(dataset_path, low_memory=False)
    print('original #records', len(df))

    # step 1 - remove scaffolding problems
    df = df[(True^df['original'].isin([0]))]
    print('after removing scaffolding problems, #records %d, #students %d' % (len(df), len(df['user_id'].unique())))

    # Step 2 - Remove problems without a skill_id
    df.dropna(subset=['skill_id'], inplace=True)
    print('after removing problems without a skill_id, #records %d, #students %d' % (len(df), len(df['user_id'].unique())))

    # Step 3 - filtered data based on max_len
    stu_seq = df.groupby('user_id', as_index=True)
    filtered_stu = [stu[0] for stu in stu_seq if len(stu[1])>max_len or len(stu[1])<min_len]
    df = df[True^df['user_id'].isin(filtered_stu)].copy()
    print('after filtered data based on min/max len, #records %d, #students %d' % (len(df), len(df['user_id'].unique())))

    # Step 4 - joint skills
    problem_seq = df.groupby('problem_id', as_index=True)
    # print('#students, ', len(df['user_id'].unique()))
    # print('#problems,', len(problem_seq))
    # print('#skills', df['skill_id'].unique().shape)
    problem_skill = {}
    joint_skill = {}
    skill_num = 0
    problem_num = 0
    for problem in problem_seq:
        skills = problem[1]['skill_id'].unique()
        tmp_skill = str(skills)
        if tmp_skill not in joint_skill:
            joint_skill[tmp_skill] = skill_num
            skill_num += 1
        problem_skill[problem[0]] = [problem_num, skill_num-1]
        problem_num += 1
    print('after joint skills, #skill', skill_num)
    
    new_problem_id = []
    new_skill_id = []
    for _, row in df.iterrows():
        user = row['user_id']
        problem = row['problem_id']
        new_problem_id.append(problem_skill[problem][0])
        new_skill_id.append(problem_skill[problem][1])
    df['new_problem_id'] = new_problem_id
    df['new_skill_id'] = new_skill_id
    df.to_csv(os.path.join(data_folder, 'new_'+csv_file))


def write_txt(file, data):
    with open(file, 'w') as f:
        for dd in data:
            for d in dd:
                f.write(str(d)+'\n')


def train_test_generator(K=5):
    t0 = time.time()
    df = pd.read_csv(os.path.join(data_folder, 'new_'+csv_file), low_memory=False)
    
    u, p, s = df['user_id'].unique(), df['new_problem_id'].unique(), df['new_skill_id'].unique()
    stu_num, pro_num, skill_num = u.shape[0], p.shape[0], s.shape[0]
    print('user idx: ', np.min(u), np.max(u), stu_num)
    print('problem idx: ', np.min(p), np.max(p), pro_num)
    print('skill idx: ', np.min(s), np.max(s), skill_num)

    ui_df = df.groupby(['user_id'], as_index=True)
    print('#user_inter', len(ui_df))

    user_inters = []
    cnt = 0
    for ui in ui_df:
        tmp_user, tmp_inter = ui[0], ui[1]
        tmp_problems = list(tmp_inter['new_problem_id'])
        tmp_skills = list(tmp_inter['new_skill_id'])
        tmp_ans = list(tmp_inter['correct'])
        user_inters.append([[len(tmp_inter), tmp_user], tmp_skills, tmp_problems, tmp_ans])
    print('user num: ', len(user_inters))

    random.shuffle(user_inters)

    N = int(math.ceil(len(user_inters) / float(K)))
    for i in range(0, len(user_inters), N):
        train_data = user_inters[:i] + user_inters[i+N:]
        test_data = user_inters[i:i+N]
        print('train/test data length', len(train_data), len(test_data))
        train_file = os.path.join(data_folder, 'train%d.txt'%(i//N))
        test_file = os.path.join(data_folder, 'test%d.txt'%(i//N))
        write_txt(train_file, train_data)
        write_txt(test_file, test_data)
    print('time used: ', time.time()-t0)



# pre_assisment()
train_test_generator(K=5)
