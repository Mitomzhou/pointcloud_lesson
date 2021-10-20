import argparse

import os
import glob
import random

import struct

import numpy as np
# Open3D:
import open3d as o3d
# PCL utils:
import pcl
from utils.segmenter import GroundSegmenter
# sklearn:
from sklearn.cluster import DBSCAN

from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def ground_segmentation(data):
    """
    Segment ground plane from Velodyne measurement
    Parameters
    ----------
    data: numpy.ndarray
        Velodyne measurements as N-by-3 numpy.ndarray
    Returns
    ----------
    segmented_cloud: numpy.ndarray
        Segmented surrounding objects as N-by-3 numpy.ndarray
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray
    """
    # TODO 01 -- ground segmentation
    N, _ = data.shape

    #
    # pre-processing: filter by surface normals
    #
    # first, filter by surface normal
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(data)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=9
        )
    )

    # keep points whose surface normal is approximate to z-axis for ground plane segementation:
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_z = np.abs(normals[:, 2])
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/6)
    downsampled = data[idx_downsampled]

    #
    # plane segmentation with RANSAC
    #
    # ground segmentation using PLANE RANSAC from PCL:
    cloud = pcl.PointCloud()
    cloud.from_array(downsampled)
    ground_segmenter = GroundSegmenter(cloud=cloud)
    inliers, model = ground_segmenter.segment()

    #
    # post-processing: get ground output by distance to segemented plane
    #
    distance_to_ground = np.abs(
        np.dot(data,np.asarray(model[:3])) + model[3]
    )

    idx_ground = distance_to_ground <= ground_segmenter.get_max_distance()
    idx_segmented = np.logical_not(idx_ground)

    segmented_cloud = data[idx_segmented]
    segmented_ground = data[idx_ground]

    print(
        f'[Ground Segmentation]: \n\tnum. origin measurements: {N}\n\tnum. segmented cloud: {segmented_cloud.shape[0]}\n\tnum. segmented ground: {segmented_ground.shape[0]}\n'
    )
    return segmented_cloud, segmented_ground



def main():
    bin_path='../data/000004.bin'
    # 获取点云对象

    pc_array = read_velodyne_bin(bin_path)
    ground_segmentation(pc_array)