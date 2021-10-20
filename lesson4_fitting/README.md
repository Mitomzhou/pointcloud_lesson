## 算法流程
1. 计算所有顶点法向量。
2. 将顶点法向量与竖直方向夹角在阈值(30度角)内的点保留，作为地面RANSAC的输入
3. RANSAC拟合平面，点到平面的距离在阈值内视为地面点(取0.3m)，其余点为聚类的点云输入。
4. 用DBSCAN聚类算法得到聚类index，再显示。
#### 原始点云
![original_pc](https://github.com/Mitomzhou/pointcloud_lesson/blob/master/imgs/lesson4/clustering_input.png)
#### 聚类结果点云
![clustering_result](https://github.com/Mitomzhou/pointcloud_lesson/blob/master/imgs/lesson4/clustering_result.png)

