# 三维点云处理课程代码
### 一、环境安装
~~~bash
sudo pip3 install open3d
sudo pip3 install pyntcloud
~~~
### 二、课程作业
#### 第1章 Introduction and Basic Algorithms
1. Build dataset for Lecture 1
    - Download ModelNet40 dataset
    - Select one point cloud from each category
2. Perform PCA for the 40 objects, visualize it.
3. Perform surface normal estimation for each point of each object, visualize it.
4. Downsample each object using voxel grid downsampling (exact, both centroid &
random). Visualize the results.
5. Write your own code, DO NOT call apis (PCL, open3d, etc.) except for