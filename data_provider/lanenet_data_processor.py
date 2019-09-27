#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的数据解析类
"""
import os.path as ops

import cv2
import numpy as np
from config import global_config

CFG = global_config.cfg

try:
    from cv2 import cv2
except ImportError:
    pass


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file):
        """

        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_binary_list, \
        self._gt_label_instance_list = self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """
        读取列表
        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_binary_list = []
        gt_label_instance_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)
        root_dir = ops.dirname(ops.abspath(dataset_info_file))
        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                gt_img_list.append(ops.join(root_dir,info_tmp[0]))
                gt_label_binary_list.append(ops.join(root_dir,info_tmp[1]))
                gt_label_instance_list.append(ops.join(root_dir,info_tmp[2]))
                
        self._dataset_size = len(gt_img_list)
        return gt_img_list, gt_label_binary_list, gt_label_instance_list

    def _random_dataset(self):
        """
        shuffle 列表
        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_binary_list = []
        new_gt_label_instance_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_binary_list.append(self._gt_label_binary_list[index])
            new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_binary_list = new_gt_label_binary_list
        self._gt_label_instance_list = new_gt_label_instance_list

    def next_batch(self, batch_size, net_flag='vgg'):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_start == 0 and idx_end > len(self._gt_label_binary_list):
            raise ValueError('Batch size不能大于样本的总数量')

        if idx_end > len(self._gt_label_binary_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size) # 最后的样本可能会训不到
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_binary_list = self._gt_label_binary_list[idx_start:idx_end]
            gt_label_instance_list = self._gt_label_instance_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels_binary = []
            gt_labels_instance = []

            for gt_img_path in gt_img_list:
                gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
                gt_img = cv2.resize(gt_img, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT))
                gt_imgs.append(gt_img) # 256x512x3
            for gt_label_path in gt_label_binary_list:
                label_img = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE)
                label_img = label_img / 255
                label_binary = cv2.resize(label_img, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                label_binary = np.expand_dims(label_binary, axis=-1)
                gt_labels_binary.append(label_binary) # 0/1 矩阵 256x512x1

            for gt_label_path in gt_label_instance_list:
                label_img = cv2.imread(gt_label_path, cv2.IMREAD_UNCHANGED)
                label_img = cv2.resize(label_img, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                gt_labels_instance.append(label_img) # 灰度图 256x512

            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels_binary, gt_labels_instance


if __name__ == '__main__':
    val = DataSet('H:/Other_DataSets/TuSimple/TuSimple/val.txt')
    b1, b2, b3 = val.next_batch(50)
    c1, c2, c3 = val.next_batch(50)
    dd, d2, d3 = val.next_batch(50)
