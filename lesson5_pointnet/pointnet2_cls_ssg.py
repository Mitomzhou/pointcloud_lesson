# Author: Mitom
# Date  : 10/28/21
# File  : pointnet2_cls_ssg
# Description: SSG(Single-scale Grouping) pointnet++分类

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import SetAbstraction


class PointNet2Cls(nn.Module):
    """
    point++分类网络
    """
    def __init__(self, num_class, normal_channel=True):
        super(PointNet2Cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64,64,128], group_all=False)
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128+3, mlp=[128,128,256], group_all=False)
        # group_all=True 所有点全局特征，不需要npoint, radius, nsample参数
        self.sa3 = SetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
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
    classifier = PointNet2Cls(10)
    x, y = classifier(input)
    print(x.shape)
    print(y.shape)