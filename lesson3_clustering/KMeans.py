# 文件功能： 实现 K-Means 算法

import numpy as np
import math
import random

class K_Means(object):
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        """
        :param n_clusters: 聚类数
        :param tolerance: 中心点误差
        :param max_iter: 最大迭代次数
        """
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        """
        获取聚类中心
        :param data: 给定聚类对象数据
        :return:
        """

        # =====初始化
        # 中心点
        cluster_centers = {}
        # 聚类后的点
        cluster_points = {}
        for i in range(self.k_):
            cluster_centers[i] = []
            cluster_points[i] = []

        # 随机选取聚类中心
        x_max = data[:, 0].max()
        x_min = data[:, 0].min()
        y_max = data[:, 1].max()
        y_min = data[:, 1].min()
        for i in range(self.k_):
            cluster_centers[i].append([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])

        print(cluster_centers)

        for iter in range(self.max_iter_):
            for point in data:
                distance = []
                for clt_center_idx in range(self.k_):
                    distance.append(euclidean_distance(cluster_centers[clt_center_idx], point))
                min_d_index = distance.index(min(distance))
                cluster_points[min_d_index].append(point)

            # 保存之前的聚类中心
            old_cluster_centers = cluster_centers
            print(old_cluster_centers)

            print(old_cluster_centers[0][0][0])

            # 均值获取中心点
            for clt_idx in range(self.k_):
                cluster_centers[clt_idx] = mean_points(cluster_points[clt_idx])
                # print(cluster_centers[clt_idx])

            # 对比中心点误差,看是否需要再优化
            optimized = True
            for clt_idx in range(self.k_):
                x_offset = (cluster_centers[clt_idx][0][0] - old_cluster_centers[clt_idx][0][0]) / old_cluster_centers[clt_idx][0][0]
                y_offset = (cluster_centers[clt_idx][0][1] - old_cluster_centers[clt_idx][0][1]) / old_cluster_centers[clt_idx][0][1]
                if x_offset * 100 > self.tolerance_ or y_offset * 100 > self.tolerance_:
                    optimized = False

            if optimized:
                break


    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始

        # 屏蔽结束
        return result


def euclidean_distance(point1, point2):
    point1 = point1[0]
    return math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))


def mean_points(points):
    x_sum = 0
    y_sum = 0
    for p in points:
        x_sum += p[0]
        y_sum += p[1]
    return [x_sum / len(points), y_sum / len(points)]



if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

