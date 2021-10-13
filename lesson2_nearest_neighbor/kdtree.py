# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet

#
class Node:
    """
    Node类，Node是tree的基本组成元素
    """
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output


def sort_key_by_vale(key, value):
    """
    构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
    :param key: 键
    :param value: 值
    :return:
        key_sorted：排序后的键
        value_sorted：排序后的值
    """
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1


def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    """
    通过递归的方式构建树
    :param root: 树的根节点
    :param db: 点云数据
    :param point_indices: 排序后的键
    :param axis: 轴
    :param leaf_size: scalar
    :return:
        root: 即构建完成的树
    """
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        
        # 作业1
        # 屏蔽开始

        # 屏蔽结束
    return root


def traverse_kdtree(root: Node, depth, max_depth):
    """
    翻转一个kd树
    :param root: kd树
    :param depth: 当前深度
    :param max_depth: 最大深度
    :return:
    """
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


def kdtree_construction(db_np, leaf_size):
    """
    构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
    :param db_np: 原始数据
    :param leaf_size: scale
    :return:
        root：构建完成的kd树
    """
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    """
    通过kd树实现knn搜索，即找出最近的k个近邻
    :param root: kd树
    :param db: 原始数据
    :param result_set: 搜索结果
    :param query: 索引信息
    :return:
        搜索失败则返回False
    """
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始

    # 屏蔽结束

    return False


def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    """
    通过kd树实现radius搜索，即找出距离radius以内的近邻
    :param root: kd树
    :param db: 原始数
    :param result_set: 搜索结果
    :param query: 索引信息
    :return:
        搜索失败则返回False
    """
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始

    # 屏蔽结束

    return False



def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    db_np = np.random.rand(db_size, dim)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()