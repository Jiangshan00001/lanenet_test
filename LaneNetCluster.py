#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中实例分割的聚类部分
"""
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
import time
import warnings
import cv2


class LaneNetCluster(object):
    """
    实例分割聚类器
    """

    def __init__(self):
        """

        """
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]


    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        db = DBSCAN(eps=0.7, min_samples=200).fit(prediction)
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        unique_labels = [tmp for tmp in unique_labels if tmp != -1]
        print('聚类簇个数为: {:d}'.format(len(unique_labels)))

        if -1 in unique_labels:
            #-1代表未实现聚类的位置
            unique_labels.remove(-1)
        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        return num_clusters, db_labels, cluster_centers

    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        通过二值分割掩码图在实例分割图上获取所有车道线的特征向量
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        print(binary_seg_ret.shape, instance_seg_ret.shape)
        idx = np.where(binary_seg_ret == 1)

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[:, idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])
        print(len(idx[0]))
        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):
        """
        过滤实例车道线位置坐标点,假设车道线是连续的, 因此车道线点的坐标变换应该是平滑变化的不应该出现跳变
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        车道线多项式拟合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                f1 = np.polyfit(y, x, 3)
                p1 = np.poly1d(f1)
                y_min = int(np.min(y))
                y_max = int(np.max(y))
                y_fit = []
                for i in range(y_min, y_max + 1):
                    y_fit.append(i)
                x_fit = p1(y_fit)
            except Warning as e:
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)


    def get_lane_lines(self, binary_logits, inst_logits):
        """

        :param binary_logits: [2x MN]
        :param inst_logits:[5x MN]
        :return:
        """
        pass
        rect1=binary_logits[0]>3.5
        rect2=binary_logits[1]<-3.5
        rect_bin = np.logical_and(rect1, rect2)
        lane_embedding_feats, lane_coordinate = self._get_lane_area(rect_bin, inst_logits)

        num_clusters, labels, cluster_centers = self._cluster_v2(lane_embedding_feats)
        # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来
        if num_clusters > 8:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:8]]
        else:
            cluster_index = range(num_clusters)

        mask_image = np.zeros((binary_logits.shape[1],binary_logits.shape[2],3), dtype=np.uint8)
        lane_lines=[]
        for index, i in enumerate(cluster_index):
            idx = np.where(labels == i)
            print(len(idx[0]))
            coord = lane_coordinate[idx]
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))
            coord = np.array(coord)
            mask_image[coord[:, 0], coord[:, 1], :] = color
            lane_lines.append([coord[:, 0], coord[:, 1]])

        return mask_image, lane_lines, cluster_index


if __name__ == '__main__':
    binary_seg_image = cv2.imread('binary_ret.png', cv2.IMREAD_GRAYSCALE)
    binary_seg_image[np.where(binary_seg_image == 255)] = 1
    instance_seg_image = cv2.imread('instance_ret.png', cv2.IMREAD_UNCHANGED)
    ele_mex = np.max(instance_seg_image, axis=(0, 1))
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)
    embedding_image = np.array(instance_seg_image, np.uint8)
    cluster = LaneNetCluster()
    mask_image = cluster.get_lane_mask(instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image)
