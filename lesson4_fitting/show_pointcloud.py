import os
import numpy as np
import struct
import open3d

def read_velodyne_bin(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def main():
    bin_path = '../data/000038.bin'
    pcd = open3d.open3d.geometry.PointCloud()
    example = read_velodyne_bin(bin_path)
    # From numpy to Open3D
    pcd.points = open3d.open3d.utility.Vector3dVector(example)
    open3d.open3d.visualization.draw_geometries([pcd])

if __name__=="__main__":
    main()
