# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import struct
from itertools import cycle, islice
import matplotlib.pyplot as plt
import open3d as o3d


def read_velodyne_bin(path):
    """
    读取bin格式的激光点云,转为numpy
    :param path:
    :return:
    """
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


def ground_segmentation(data):
    """
    从点云文件中滤除地面点
    :param data: 一帧完整点云
    :return: segmengted_cloud: 删除地面点之后的点云
             segmengted_ground: 地面点云
    """

    # 1、保留顶点法向量与Z轴方向上的夹角小于阈值的点(PI/6)
    # 点云对象
    pcd_original = o3d.geometry.PointCloud()
    # numpy 转 open3d点
    pcd_original.points = o3d.utility.Vector3dVector(data)
    print(pcd_original.points)

    # 去除离群点 TODO
    # _, select_index = pcd_original.remove_statistical_outlier(nb_neighbors=20, std_ratio=1)
    # pcd_original.select_by_index(select_index)

    # 顶点法向量 (radius=5.0, max_nn=10，搜索半径和最大最近邻居)
    pcd_original.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=10))

    # 保留与竖直方向的法向量 open3d转numpy
    normals = np.asarray(pcd_original.normals)
    # 顶点法向量与Z轴方向上的夹角
    angular_z = np.abs(normals[:,2])
    index_downsampled = angular_z > np.cos(np.pi/6)
    downsampled = data[index_downsampled]

    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(downsampled)

    # 2、平面分割得到平面参数
    # segement_plane: 这个函数返回（a, b, c, d）作为一个平面，对于平面上每个点(x, y, z) ax + by + cz + d = 0, 返回[a,b,c,d]
    #   destance_threshold: 点到一个估计平面的最大距离，这些距离内的点被认为是内点（inlier）
    #   ransac_n: 随机抽样估计一个平面的点的个数
    #   num_iterations: 随机平面采样和验证的频率（迭代次数）
    # Return:
    #   plane_model: [a,b,c,d]
    #   inliners: 平面上的点index
    distance_threshold = 0.3
    plane_model, inliners = pcd_downsampled.segment_plane(distance_threshold=distance_threshold, ransac_n=5,num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 点到平面距离
    distance_to_ground = np.abs(np.dot(data, np.asarray(plane_model[:3])) + plane_model[3])
    # 点到平面距离小于等于distance_threshold视为地面点
    index_ground = distance_to_ground <= distance_threshold
    # 其他为非地面点云
    index_segmented = np.logical_not(index_ground)

    segmengted_cloud, segmengted_ground = data[index_segmented], data[index_ground]

    print('初始点云数目:', data.shape[0])
    print('分割后点云数目:', segmengted_cloud.shape[0])
    print('地面点云数目:', segmengted_ground.shape[0])
    return segmengted_cloud, segmengted_ground


def clustering(data):
    """
    从点云中提取聚类
    :param data: 点云（滤除地面后的点云）
    :return:
        clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
    """
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(data)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        clusters_index = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10, print_progress=True))

    return clusters_index


def plot_clusters(data, cluster_index):
    """
    显示聚类点云，每个聚类一种颜色
    :param data: data：点云数据（滤除地面后的点云）
    :param cluster_index: cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
    :return:
    """
    max_label = cluster_index.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(cluster_index / (max_label if max_label > 0 else 1))

    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(data)

    colors[cluster_index < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd], "Open3D dbscanclusting")


def main():
    bin_path = '../data/000038.bin'
    pcd_array = read_velodyne_bin(bin_path)


    segmengted_cloud, segmengted_ground = ground_segmentation(pcd_array)
    clusters_index = clustering(segmengted_cloud)
    plot_clusters(segmengted_cloud, clusters_index)


if __name__=="__main__":
    main()
