# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
 
color = np.array(['blue', 'black', 'gray'])
 
def plot_cluster(data, cls=None, cluster=None, title=''):
    if cls is None:
        c = [color[0]] * data.shape[0]
    else:
        c = color[cls].tolist()
    plt.scatter(data[:, 0], data[:, 1], s=150, c=c)
    # for i, clu in enumerate(cluster):
    plt.scatter(cluster[:, 0], cluster[:, 1], s=180, c='red',marker='*')
    plt.title(title)
    plt.show()
    plt.close()
 
def distance(data, cluster):
    return np.sum(np.power(data[:, None] - cluster[None], 2), axis=-1)
 
def k_means(data, k):
 
    # 随机选取K个数据作为簇心
    cluster = data[np.random.choice(data.shape[0], k, replace=False)]
    print(f'init cluster: {cluster}')
 
    # 构建与data.shape[0]相同长度一维的数据，用于记录每个元素属于哪个簇心是否需要更改
    last_loc = np.zeros(data.shape[0])
    step = 0
    plot_cluster(data, cls=None, cluster=cluster, title=f'step: {step}')
 
    while True:
        # 计算每个坐标点与随机选取的簇心之间的欧式距离/1-IOU距离
        d = distance(data, cluster)
        # 选取每个坐标点与K个簇心距离最小的那个，则该坐标归属于距离最小的簇心
        current_loc = np.argmin(d, axis=-1)
        # 如果这两者完全相等，或者差异小于一定范围（这里范围是10，如果不设置，那么1-wh_iou则执行几百次还没找到），则说明没有坐标需要更改，则簇心更新完
        if (last_loc == current_loc).all() or np.sum(last_loc == current_loc) >= np.sum(np.ones(data.shape[0]-10)):
            break
        # 计算每个簇内数据的中值，作为每个簇新的簇心
        for clu in range(k):
            print('current_loc')
            print(current_loc)
            print('clu')
            print(clu)
            print(current_loc == clu)
            cluster[clu] = np.median(data[current_loc == clu], axis=0)
        last_loc = current_loc
        step += 1
        plot_cluster(data, cls=current_loc, cluster=cluster, title=f'step: {step}')
    print(f'step: {step}')
    print(f'final cluster: {cluster}')
    return cluster
 
 
def wh_iou(wh1, wh2):
    wh1 = wh1[:,None]
    wh2 = wh2[None]
    inter = np.minimum(wh1, wh2).prod(2)
 
    iou = inter/(wh1.prod(1)+wh2.prod(1)-inter)
    return iou
 
if __name__ == '__main__':
    # 这里先创建一些点符合设定的正泰分布的数据，然后把这些构建为坐标点（坐标为x,y）
    x1 = np.random.normal(loc=1, size=180)
    y1 = np.random.normal(loc=3, size=180)
    # 构建为坐标点
    data = np.concatenate([x1[:, None], y1[:, None]], axis=-1)
    k_means(data, k=3)
 