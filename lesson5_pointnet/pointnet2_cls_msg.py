# Author: Mitom
# Date  : 10/28/21
# File  : pointnet2_cls_msg
# Description:  MSG(Multi-scale Grouping) PointNet++分类

import datetime
import os.path
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import SetAbstraction, SetAbstractionMsg


class PointNet2ClsMsg(nn.Module):
    """
    MSG(Multi-scale Grouping) PointNet++分类
    """
    def __init__(self, num_class, normal_channel=True):
        super(PointNet2ClsMsg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = SetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128], in_channel=in_channel, mlp_list=[[32,32,64],[64,64,128],[64,96,128]])
        self.sa2 = SetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128], in_channel=320, mlp_list=[[64,64,128],[128,128,256],[128,128,256]])
        # group_all=True 所有点全局特征，不需要npoint, radius, nsample参数
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+256+256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)


    def forward(self, xyz):
        """
        :param xyz: (B, C, N) C=3
        :return:
            x: (B, num_class)
            sal3_new_points: (B, last_mlp[-1]=1024, 1)
        """
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        sal1_new_xyz, sal1_new_points = self.sa1(xyz, norm)
        print("sal1_new_xyz", sal1_new_xyz.shape, "sal1_new_points", sal1_new_points.shape)

        sal2_new_xyz, sal2_new_points = self.sa2(sal1_new_xyz, sal1_new_points)
        print("sal2_new_xyz", sal2_new_points.shape, "sal2_new_points", sal2_new_points.shape)

        sal3_new_xyz, sal3_new_points = self.sa3(sal2_new_xyz, sal2_new_points)
        print("sal3_new_xyz", sal3_new_xyz.shape, "sal3_new_points", sal3_new_points.shape)

        x = sal3_new_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, sal3_new_points


class PointNet2ClsLoss(nn.Module):
    def __init__(self):
        super(PointNet2ClsLoss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss


if __name__ == "__main__":
    input = torch.rand(4, 6, 1000)
    classifier = PointNet2ClsMsg(10)
    x, y = classifier(input)
    print(x.shape)
    print(y.shape)




