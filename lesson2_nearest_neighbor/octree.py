# octree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet


class Octant:
    """
    节点，构成OCtree的基本元素
    """
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output


def traverse_octree(root: Octant, depth, max_depth):
    """
    翻转octree
    :param root: 构建好的octree
    :param depth: 当前深度
    :param max_depth: 最大深度
    :return:
    """
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1


def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    """
    通过递归的方式构建octree
    :param root: 根节点
    :param db: 原始数据
    :param center: 中心
    :param extent: 当前分割区间
    :param point_indices: 点的key
    :param leaf_size: scale
    :param min_extent: 最小分割区间
    :return:
    """
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        pass
        # 作业4
        # 屏蔽开始
        
        # 屏蔽结束
    return root


def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    判断当前query区间是否在octant内
    :param query: 索引信息
    :param radius: 索引半径
    :param octant: octree
    :return:
        判断结果，即True/False
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)


def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    判断当前query区间是否和octant有重叠部分
    :param query: 索引信息
    :param radius: 索引半径
    :param octant: octree
    :return:
        判断结果，即True/False
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


def contains(query: np.ndarray, radius: float, octant:Octant):
    """
    判断当前query是否包含octant
    :param query: 索引信息
    :param radius: 索引半径
    :param octant: octree
    :return:
        判断结果，即True/False
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius


def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    """
    在octree中查找信息
    :param root: octree
    :param db: 原始数据
    :param result_set: 索引结果
    :param query: 索引信息
    :return:
    """
    if root is None:
        return False

    # 作业5
    # 提示：尽量利用上面的inside、overlaps、contains等函数
    # 屏蔽开始
    
    # 屏蔽结束

    return inside(query, result_set.worstDist(), root)


def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    """
    在octree中查找radius范围内的近邻
    :param root: octree
    :param db: 原始数据
    :param result_set: 搜索结果
    :param query: 搜索信息
    :return:
    """
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业6
    # 屏蔽开始
    
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    """
    在octree中查找最近的k个近邻
    :param root: octree
    :param db: 原始数据
    :param result_set: 搜索结果
    :param query: 搜索信息
    :return:
    """
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业7
    # 屏蔽开始
    
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


def octree_construction(db_np, leaf_size, min_extent):
    """
    构建octree，即通过调用octree_recursive_build函数实现对外接口
    :param db_np: 原始数据
    :param leaf_size: scale
    :param min_extent: 最小划分区间
    :return:
    """
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = np.mean(db_np, axis=0)

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root


def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    # depth = [0]
    # max_depth = [0]
    # traverse_octree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # octree_knn_search(root, db_np, result_set, query)
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])

    begin_t = time.time()
    print("Radius search normal:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.5)
        octree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    begin_t = time.time()
    print("Radius search fast:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius = 0.5)
        octree_radius_search_fast(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t)*1000))



if __name__ == '__main__':
    main()