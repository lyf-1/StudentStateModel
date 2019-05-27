
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from matplotlib import cm 
from matplotlib import axes
plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False


def draw_heatmap(data, xlabels, ylabels, name='1.pdf'):
    cmap=cm.Blues    
    # cmap=cm.get_cmap('tab10',1000)
    figure=plt.figure(facecolor='w')
    ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels,rotation=-45)

    vmax=data[0][0]
    vmin=data[0][0]
    for i in data:
        for j in i:
            if j>vmax:
                vmax=j
            if j<vmin:
                vmin=j
    map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
    cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
    
    plt.xlabel(u'答题序列')
    plt.ylabel(u'知识点')
    plt.show()
    # plt.savefig(name)


def read_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip('\t\n').split('\t')
            line = [float(ele) for ele in line]
            data.append(line)
    data = np.array(data).astype(np.float32)
    print('data shape: ', data.shape)
    return data


# filename = 'weights.txt'
# a = read_data(filename)
# a = a[:10, :10]
# print(a.shape)

# xlabels = [(19,1),(18,0),(18,1),(20,1),(20,1),(25,1),(23,1),(22,1),(23,1),(23,1)]
# ylabels = [(19,1),(18,0),(18,1),(20,1),(20,1),(25,1),(23,1),(22,1),(23,1),(23,1)]
# # 1,0,1,1,1,1,1,1,1,1
# # xlabels = np.arange(a.shape[1])
# # ylabels = np.arange(a.shape[0])
# draw_heatmap(a, xlabels, ylabels, name=filename+'.pdf') 

q = [19,18,18,20,20,25,23,22,23,23,19,18,19,20,21,21,21,23,23,40,40,40,40,40,40,40,40,40,40,27,40,24,60,60,19,24,60,24,46,46,46]
a = [1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1]
xlabels = []
for i in range(len(q)):
    ele = '(%d,%d)' % (q[i], a[i])
    xlabels.append(ele)
action = list(set(q))
action.sort()
print(action)

with open('selfatten.pkl', 'rb') as f:
    data = pkl.load(f)
    data = data[:40,:].T
    data = data[action, :]
print(data.shape)
print(data)

# xlabels = np.arange(data.shape[1])
# ylabels = np.arange(data.shape[0])
# xlabels = q
ylabels = action
draw_heatmap(data, xlabels, ylabels, name='selfatten.pdf') 