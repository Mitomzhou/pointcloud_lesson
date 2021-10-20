import os
import numpy as np
import struct
import open3d as o3d


def read_bin_velodyne(path):
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

def main():
    bin_path='../data/000004.bin'
    # 获取点云对象
    pcd_original = o3d.open3d.geometry.PointCloud()
    pc_array = read_bin_velodyne(bin_path)

    # numpy 转 open3d
    pcd_original.points= o3d.open3d.utility.Vector3dVector(pc_array)
    print("原始点数目：", pcd_original)

    # 点云可视化
    # o3d.open3d.visualization.draw_geometries([pcd_original])

    # 去除离群点
    cl, ind = pcd_original.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)

    pcd_filtered = pcd_original.select_by_index(ind)
    # o3d.open3d.visualization.draw_geometries([pcd_filtered])
    print("去除离群点数目：", pcd_filtered)

    # 平面分割 segment_plane
    # segement_plane: 这个函数返回（a, b, c, d）作为一个平面，对于平面上每个点(x, y, z) ax + by + cz + d = 0, 返回[a,b,c,d]
    #   destance_threshold: 点到一个估计平面的最大距离，这些距离内的点被认为是内点（inlier）
    #   ransac_n: 随机抽样估计一个平面的点的个数
    #   num_iterations: 随机平面采样和验证的频率（迭代次数）
    # Return:
    #   plane_model: [a,b,c,d]
    #   inliers: 平面上的点index
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.1,ransac_n=5,num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    ground_cloud = pcd_filtered.select_by_index(inliers)
    no_ground_cloud = pcd_filtered.select_by_index(inliers, invert=True)
    # o3d.open3d.visualization.draw_geometries([ground_cloud])
    print("地面点数目：", ground_cloud)
    print("非地面点数目：", no_ground_cloud)
    o3d.open3d.visualization.draw_geometries([no_ground_cloud])


    # #print(inliers)
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    #
    # print('----outlier_cloud: ', outlier_cloud.points)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name='Open3D Plane Model', width=1920,
    #                               height=1080, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
    #                               mesh_show_back_face=False)








    # # 点云顶点法向量,(搜索半径0.1m，最多考虑30个最近邻)
    # pcd_original.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #
    #
    # # 顶点法向量可视化
    # o3d.visualization.draw_geometries([pcd_original], window_name='Open3D Normals', width=1920, height=1080,
    #                                   left=50, top=50, point_show_normal=True, mesh_show_wireframe=False,
    #                                   mesh_show_back_face=False)
    #
    # # normals
    # normals = np.asarray(pcd_original.normals)
    # print(normals)










if __name__=="__main__":
    main()
