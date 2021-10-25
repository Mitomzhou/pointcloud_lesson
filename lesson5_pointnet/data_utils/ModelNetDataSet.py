"""
ModelNet40数据集Loader
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNetDataSet(Dataset):
    """
    ModelNet40 数据集
    """

    def __init__(self, root, split='train', npoints=1024):
        """
        :param root: 数据路径
        :param split: train/test
        """
        self.root = root
        self.objlist = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_shape_names.txt'))]
        self.classes = dict(zip(self.objlist, range(len(self.objlist)))) # {'airplane': 0, 'bathtub': 1, 'bed': 2 ...}
        self.npoints = npoints

        # 加载训练和测试数据文件路径
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')

        # 标签name
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # (标签name, 数据文件路径)
        self.datapath_tuple = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

    def __len__(self):
        return len(self.datapath_tuple)

    def __getitem__(self, index):
        one_data = self.datapath_tuple[index]
        cls_idx = self.classes[self.datapath_tuple[index][0]]
        label = np.array([cls_idx]).astype(np.int32)
        point_set = np.loadtxt(one_data[1], delimiter=',').astype(np.float32)

        # 最远点采样
        point_set = farthest_point_sample(point_set, self.npoints)
        # 点云归一化
        point_set[:, 0:3] = pointcloud_normalize(point_set[:, 0:3])
        return point_set, label[0]


def farthest_point_sample(point, npoint):
    """
    最远点采样
    :param point: [N, D] N=单个点云点数目 D=深度 3/6
    :param npoint: 最远点采样的样本数
    :return: npoint index， [npoint, D]  采样的点云, npoint=1024 D=3/6
    """

    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pointcloud_normalize(pc):
    """
    点云归一化， 可以提高结果的精度， 同时， 归一化算法对于任何尺度缩放和坐标原点的选择都是不变的。
    :param pc:
    :return:
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


if __name__ == "__main__":
    c = 0
    dataset = ModelNetDataSet('/home/mitom/3DPointCloud/data/modelnet40_normal_resampled', split='test')
    train_loader = torch.utils.data.DataLoader(dataset)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(data.shape)
        print(label)
        c = c + 1
        print("------------------: ", c)
