import csv
import math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

 
def process_file(input_file, output_file, max_lines=None):
    # read learning trace data in specified format, see README for details
    print('reading data...')

    stu_item = {}
    stu_p = {}
    stu_nreps = {}
    stu_delta = {}
    lexeme_index = {}
    item_cnt = 0

    f = open(input_file, 'r')
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        
        stu = row['user_id']
        lexeme = row['lexeme_id']
        p = row['p_recall']
        nreps = int(row['history_seen'])
        # delta = row['delta']
        delta = float(row['delta'])/(60*60*24)  # convert time delta to days
        
        if lexeme not in lexeme_index:
            lexeme_index[lexeme] = item_cnt
            item_cnt += 1

        if stu in stu_item:
            stu_item[stu].append(lexeme)
            stu_p[stu].append(p)
            stu_nreps[stu].append(nreps)
            stu_delta[stu].append(delta)
        else:
            stu_item[stu] = [lexeme]
            stu_p[stu] = [p]
            stu_nreps[stu] = [nreps]
            stu_delta[stu] = [delta]

        if i % 1000000 == 0:
            print('%d...\t' % i)
    print(i)
    f.close()

    f = open('preprocess/stu_item.txt', 'w')
    f.write(str(stu_item))
    f.close
    f = open('preprocess/stu_p.txt', 'w')
    f.write(str(stu_p))
    f.close
    f = open('preprocess/stu_nreps.txt', 'w')
    f.write(str(stu_nreps))
    f.close
    f = open('preprocess/stu_delta.txt', 'w')
    f.write(str(stu_delta))
    f.close

    out = open(output_file, 'w')
    fout = csv.writer(out)
    for stu in stu_item.keys():
        fout.writerow([stu])
        fout.writerow(stu_item[stu])
        fout.writerow(stu_p[stu])
        fout.writerow(stu_nreps[stu])
        fout.writerow(stu_delta[stu])
    out.close()


def read_file(folder='preprocess/', min_len=150, max_len=250, max_num=None):
    files = ['item', 'p', 'nreps', 'delta']
    data = []
    for file in files:
        f = open(folder+'stu_'+file+'.txt', 'r')
        tmp = f.read()
        tmp = eval(tmp)
        f.close()
        data.append(tmp)
    
    stu_item, stu_p, stu_nreps, stu_delta = data
    all_items = []
    all_p = []
    all_delta = []
    all_nreps = []

    sample_num = 0
    items_num = set()
    user_item_num = 0
    for stu in stu_item.keys():
        if max_num is not None and sample_num >= max_num:
            break 
        items = stu_item[stu]  # here, items is lexeme
        if len(items) >= min_len and len(items) <= max_len:
            all_items.append(items)
            all_p.append(stu_p[stu])
            all_delta.append(stu_delta[stu])
            all_nreps.append(stu_nreps[stu])

            sample_num += 1
            user_item_num += len(items)
            for it in items:
                items_num.add(it)
            
    print('items number: ', len(items_num))
    print('samples: ', sample_num)    
    print('user item pair: ', user_item_num)
  
    dump_path = folder+'pkl/'+'duolingo_delta_%d_%d_%d.pkl' % (sample_num, min_len, max_len)
    dump_file = open(dump_path, 'wb')
    pkl.dump([all_items, all_p, all_delta, all_nreps], dump_file)



def duolingo_for_lstm(load_path, thr=0.859):
    max_len = int(load_path.split('.')[0].split('_')[-1])
    user_num = int(load_path.split('.')[0].split('_')[-3])
    lexeme, answer, delta, nreps = pkl.load(open(load_path, 'rb'))

    lexeme_idx = {}
    items_cnt = 1
    for episode in range(len(lexeme)):
        for seq_idx in range(len(lexeme[episode])):
            tmp_lexeme = lexeme[episode][seq_idx] 
            if tmp_lexeme not in lexeme_idx:
                lexeme_idx[tmp_lexeme] = items_cnt
                items_cnt += 1                
    # print('items number: ', items_cnt-1, len(lexeme_idx))
    items_cnt = len(lexeme_idx)
    q_data = []
    qa_data = []
    targets = []
    delta_data = []
    nreps_data = []
    p_recall = []

    num_true = 0
    num_false = 0
    num_all = 0
    for episode in range(len(lexeme)):
        q_episode = [0] * max_len
        qa_episode = [0] * max_len
        # targets_episode = [-1] * max_len
        delta_episode = [0] * max_len
        nreps_episode = [0] * max_len
        # p_episode = [-1] * max_len
        for seq_idx in range(len(lexeme[episode])):
            tmp_lexeme = lexeme[episode][seq_idx] 
            act = lexeme_idx[tmp_lexeme] 
            ans = 1 if float(answer[episode][seq_idx]) >= thr else 0 

            if ans == 1:
                num_true += 1
            if ans == 0:
                num_false += 1
            num_all += 1

            q_episode[seq_idx] = act
            qa_episode[seq_idx] = act + ans * items_cnt
            # targets_episode[seq_idx] = ans
            delta_episode[seq_idx] = delta[episode][seq_idx]
            nreps_episode[seq_idx] = nreps[episode][seq_idx]
            # p_episode[seq_idx] = float(answer[episode][seq_idx])
        q_data.append(q_episode)
        qa_data.append(qa_episode)
        # targets.append(targets_episode)
        delta_data.append(delta_episode)
        nreps_data.append(nreps_episode)
        # p_recall.append(p_episode)
    
    q_data = np.array(q_data).astype(np.int32)
    qa_data = np.array(qa_data).astype(np.int32)
    # targets = np.array(targets).astype(np.float32)
    delta_data = np.array(delta_data).astype(np.float32)
    nreps_data = np.array(nreps_data).astype(np.float32)
    # p_recall = np.array(p_recall).astype(np.float32)
    print('q, qa shape: ', q_data.shape, qa_data.shape)

    print(num_true, num_false)
    print(float(num_true)/num_all)
    print(float(num_true)/(num_true+num_false))
    
    # # return q_data, qa_data, targets, delta_data
    dump_path = 'preprocess/feature/duolingo_%d_%d_%d.pkl'%(user_num, max_len, len(lexeme_idx))
    with open(dump_path, 'wb') as file:
        pkl.dump([q_data, qa_data, delta_data, nreps_data], file)


if __name__ == "__main__":
    # process_file('data/learning_traces.13m.csv', 'preprocess/duolingo.csv', max_lines=None)
    # read_file(min_len=30, max_len=500, max_num=20000)
    
    path = 'preprocess/pkl/duolingo_delta_20000_30_500.pkl'
    duolingo_for_lstm(path)
