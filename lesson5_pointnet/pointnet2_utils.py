# Author: Mitom
# Date  : 
# File  : pointnet2_utils
# Description: PointNet++ SetAbstraction模块

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAbstraction(nn.Module):
    """
    set abstraction (for Single-scale Grouping)
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        """
        :param npoint: 中心点个数,FPS取的sampling, SA1为512， SA2为128
        :param radius: 搜索球半径, 主导
        :param nsample: 局部区域最大点数目，球内最大点数
        :param in_channel: 输入通道数(上一个SA层的out_channel + 3) = last_mlp[-1] + 3
        :param mlp: 全连接层的节点数，如3层[64,64,128]
        :param group_all: 局部group/全局group
        """
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all


    def forward(self, xyz, points):
        """
        :param xyz: point的坐标信息 (B,C,N)  B=batch_num, C=3, N=点数
        :param points: 除点坐标信息外的信息即点集特征 (B,D,N)
        :return:
            new_xyz: (B,C,S) C=3  S=npoint
            new_points: (B,D',S)
        """
        xyz = xyz.permute(0, 2, 1)  # (B,C,N) -> (B,N,C)
        if points is not None:
            points = points.permute(0, 2, 1)    # (B,D,N) -> (B,N,D)
        # group过程
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz:(B,npoint,nsample,3), new_points: (B,npoint,nsample,3+D)
        new_points = new_points.permute(0, 3, 2, 1)  # (B,3+D,nsample,npoint)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class SetAbstractionMsg(nn.Module):
    """
    set abstraction (for Multi-scale Grouping)
    """
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """
        :param npoint: FPS采样点
        :param radius_list: eg. [0.1, 0.2, 0.4]
        :param nsample_list: eg. [16, 32, 128]
        :param in_channel:
        :param mlp_list: [[32,32,64], [64,64,128],[64,96,128]]
        """
        super(SetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        # 初始化 conv+bn blocks
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        :param xyz: 点坐标信息 (B,3,N)
        :param points: 点其他信息 (B,D,N)
        :return:
            new_xyz: 采样点位置信息 (B,3,S)
            new_points_concat: 采样点feature (B,D',S)
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    :param xyz: （B,N,3）
    :param npoint: 采样点数
    :return:
        centroids: 采样后点的index (B, npoint)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    主要用于将整个点云分散成局部的group，对每一个group都可以用PointNet单独的提取局部的全局特征。
    步骤：
        1. 先用farthest_point_sample函数实现最远点采样FPS得到采样点的索引，再通过index_points将这些点的从原始点中挑出来，作为new_xyz。
        2. 利用query_ball_point和index_points将原始点云，通过new_xyz 作为中心分为npoint个球形区域，其中每个区域有nsample个采样点。
        3. 每个区域的点减去区域的中心值（归一化处理）。
        4. 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征。
    代码理解参考：https://blog.csdn.net/weixin_42707080/article/details/105307595?spm=1001.2014.3001.5501
    :param npoint: 中心点个数,FPS取的sampling
    :param radius: 搜索半径
    :param nsample: 局部区域点数目
    :param xyz: 点坐标信息 (B,N,3)
    :param points: 除点坐标信息外的信息即点集特征 (B,D,N)
    :param returnfps: 是否返回FPS点
    :return:
        new_xyz: 采样点坐标信息 (B,npoint,nsample,3)
        new_points: 采样点数据信息(包含坐标信息) (B,npoint,nsample,3+D)  nsample=C  npoint=S
    """

    B, N, C = xyz.shape
    S = npoint
    # FPS采样后得到的index (B, npoint)，
    fps_idx = farthest_point_sample(xyz, npoint)
    # 索引点 (B,S,3）S=npoint
    new_xyz = index_points(xyz, fps_idx) # (B,S,3)
    # (B,S,nsample) S=npoint
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # group的坐标信息 (B,npoint,nsample,C)
    grouped_xyz = index_points(xyz, group_idx)
    # group的特征信息（过程中减去中心点，理解为归一化norm）
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, group_idx)
        # 拼接group坐标信息与局部特征 # (B,npoint,nsample,3+D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    直接将所有点作为一个group
    :param xyz: 点坐标信息 (B,N,3)
    :param points: 除点坐标信息外的信息即点集特征 (B,D,N)
    :return:
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points



def index_points(points, idx):
    """
    根据(B,S)对(B,N,C)索引点数据，赋值到new_points
    :param points: 输入点数据 (B,N,C)
    :param idx: FPS采样后的index，(B,S)
    :return:
        new_points: 索引后的点数据 (B,S,C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    计算两个点群对象的欧式距离
    :param src: (B,N,C)
    :param dst: (B,M,C)
    :return:
        (B,N,M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    在radius内搜索点，radius主导，上限是nsample个点
    代码理解参考：https://blog.csdn.net/weixin_42707080/article/details/105302517?spm=1001.2014.3001.5501
    :param radius: 搜索半径
    :param nsample: 局部区域最大点数
    :param xyz: 所有点 (B,N,3)
    :param new_xyz: 初始中心点 (B,S,3)
    :return:
        group_idx: (B,S,nsample) S=npoint
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


if __name__ == "__main__":
    npoint = 10
    radius = 0.2
    nsample = 5
    in_channel = 7  # xyz.shape[1] + points.shape[1]
    mlp = [4,8,16]
    group_all = False

    xyz = torch.rand(2,3,100)
    points = torch.rand(2, 4, 100)
    sa1 = SetAbstraction(npoint, radius, nsample, in_channel, mlp, group_all)
    new_xyz, new_points = sa1(xyz, points)
    print(new_xyz.shape)
    print(new_points.shape)
    print(new_xyz)
    print(new_points)




