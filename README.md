# 三维点云处理课程代码
### 一、环境安装
3D工具包
~~~bash
sudo pip3 install open3d
sudo pip3 install pyntcloud
~~~
离线安装pytorch-1.8.0+cu111 [[离线下载]](https://download.pytorch.org/whl/torch/)
~~~bash
sudo pip3 install torch-1.8.0+cu111-cp38-cp38-linux_x86_64.whl -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
~~~
离线安装tensorflow-gpu_2.6.0 [[离线下载]](https://pypi.org/project/tensorflow-gpu/2.6.0/#files)
~~~bash
sudo pip3 install tensorflow_gpu-2.6.0-cp38-cp38-manylinux2010_x86_64.whl -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
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