# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False


def read_logs(path):
    auc = []
    with open(path, 'r') as f:
        for line in f:
            line = float(line.split(':')[2].split(',')[0])
            auc.append(line)
    return auc


def plot_auc(auc):
    x = np.arange(len(auc))
    
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'AUC')
    plt.plot(x, auc)
    plt.show()


file = 'logs/DKVMN_1558075445.log'
auc_data = read_logs(file)
plot_auc(auc_data)