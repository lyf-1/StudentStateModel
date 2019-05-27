import numpy as np
import random
import copy
import csv


random.seed(0)


class DataGenerator():
    def __init__(self, input_file, skill_num, max_len, shuffle_flag=False):
        """
        self.a, self.b, self.c, self.d:
            (qa, next_q, ans, hist_q) # (int32, int32, float32, int32)
        """

        self.batch_id = 0
        self.skill_num = skill_num
        self.max_len = max_len
        self.shuffle_flag = shuffle_flag
        self.a, self.b, self.c, self.d = [], [], [], []

        rows = []
        with open(input_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                rows.append(row)
        
        index = 0
        print("the number of rows is " + str(len(rows)))
        min_id, max_id = 100000000000, -1
        tmp_max_len = -1
        while(index < len(rows)-1):
            num = int(rows[index][0])
            tmp_skills = list(map(int, rows[index+1]))[:max_len+1]
            tmp_ans = list(map(int, rows[index+2]))[:max_len+1]
            if(num > 2):
                tmp_a = []
                for j in range(len(tmp_skills)):
                    tmp_a.append(tmp_skills[j]+self.skill_num*tmp_ans[j])
                tmp_b = tmp_skills
                
                tmp_max_len = max(tmp_max_len, len(tmp_a[:-1]))
                min_id = min(min(tmp_b), min_id)
                max_id = max(max(tmp_b), max_id)

                self.a.append(tmp_a[:-1])
                self.b.append(tmp_b[1:])
                self.c.append(tmp_ans[1:])
                self.d.append(tmp_b[:-1])
                
            index += 3
        
        assert max_len == tmp_max_len
        print('max seq len, ', max_len, tmp_max_len)
        print('min / max skill id', min_id, max_id)
        
        self.seq_num = len(self.a)
        if self.shuffle_flag:
            self.shuffle()
    
    def shuffle(self):
        data = list(zip(self.a, self.b, self.c, self.d))
        random.shuffle(data)
        self.a[:], self.b[:], self.c[:], self.d[:] = zip(*data)

    def next(self, batch_size):
        if self.batch_id >= self.seq_num:
            if self.shuffle_flag:
                self.shuffle()
            self.batch_id = 0
        batch_a = copy.deepcopy(self.a[self.batch_id:self.batch_id+batch_size])
        batch_b = copy.deepcopy(self.b[self.batch_id:self.batch_id+batch_size])
        batch_c = copy.deepcopy(self.c[self.batch_id:self.batch_id+batch_size])
        batch_d = copy.deepcopy(self.d[self.batch_id:self.batch_id+batch_size])
        self.batch_id += batch_size

        batch_real_seq_len = []
        for i in range(len(batch_a)):
            pad_a, pad_b, pad_c, pad_d = 0, 0, -1, 0
            cur_len = len(batch_a[i])
            batch_real_seq_len.append(cur_len)
            for j in range(self.max_len-cur_len):
                batch_a[i].append(pad_a)
                batch_b[i].append(pad_b)
                batch_c[i].append(pad_c)
                batch_d[i].append(pad_d)
                
        batch_a = np.array(batch_a).astype(np.int32)
        batch_b = np.array(batch_b).astype(np.int32)
        batch_c = np.array(batch_c).astype(np.float32)
        batch_d = np.array(batch_d).astype(np.int32)
        batch_real_seq_len = np.array(batch_real_seq_len).astype(np.int32)
        return batch_a, batch_b, batch_c, batch_d, batch_real_seq_len

