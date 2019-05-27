import numpy as np


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


def pass_rate(ans):
    num_pass = np.sum(ans==1)
    num_fail = np.sum(ans==0)
    print('num pass / num fail / num all', num_pass, num_fail, ans.shape)
    return float(num_pass) / (num_fail+num_pass)
