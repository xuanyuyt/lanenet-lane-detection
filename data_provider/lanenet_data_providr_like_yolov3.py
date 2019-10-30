#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
# @Time    : 2019/10/29 16:44
# @Author  : YangTao
# @Site    : 
# @File    : lanenet_data_providr_like_yolov3.py
# @IDE: PyCharm Community Edition
# ================================================================
"""
实现LaneNet的数据解析类
"""
import os.path as ops
import cv2
import numpy as np
from config.global_config import cfg
import tensorflow as tf

class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_type, net_flag='mobilenet_v2'):
        """

        :param dataset_info_file:
        """
        self.net_flag = net_flag
        self.lane_path = cfg.TRAIN.LANE_PATH if dataset_type == 'train' else cfg.TEST.LANE_PATH
        self._gt_img_list, self._gt_label_binary_list, \
        self._gt_label_instance_list = self._init_dataset(self.lane_path)
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)
        self._random_dataset() # shuffle 列表
        self.train_input_size_h = cfg.TRAIN.IMG_HEIGHT
        self.train_input_size_w = cfg.TRAIN.IMG_WIDTH
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        if self.batch_size > self._dataset_size:
            raise ValueError('Batch size不能大于样本的总数量')
        self.num_batchs = int(np.ceil(self._dataset_size / self.batch_size))  # 向上取整
        self.batch_count = 0

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

                gt_img_list.append(ops.join(root_dir, info_tmp[0]))
                gt_label_binary_list.append(ops.join(root_dir, info_tmp[1]))
                gt_label_instance_list.append(ops.join(root_dir, info_tmp[2]))

        self._dataset_size = len(gt_img_list)
        return gt_img_list, gt_label_binary_list, gt_label_instance_list

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batchs

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

    def __next__(self):
        with tf.device('/cpu:0'):
            batch_image = np.zeros((self.batch_size, self.train_input_size_h, self.train_input_size_w, 3))
            batch_labels_binary = np.zeros((self.batch_size, self.train_input_size_h, self.train_input_size_w, 1))
            batch_labels_instance = np.zeros((self.batch_size, self.train_input_size_h, self.train_input_size_w, 1))

            num = 0  # sample in one batch's index
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self._dataset_size:  # 从头开始
                        index -= self._dataset_size
                    gt_img_path = self._gt_img_list[index]
                    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
                    gt_img = cv2.resize(gt_img, (self.train_input_size_w, self.train_input_size_h))
                    if self.net_flag == 'vgg':
                        gt_img = gt_img / 127.5 - 1.0  # 归一化到 -1,1
                    elif self.net_flag == 'mobilenet_v2':
                        gt_img = gt_img - [103.939, 116.779, 123.68]
                    batch_image[num, :, :, :] = gt_img

                    gt_label_path = self._gt_label_binary_list[index]
                    label_img = cv2.imread(gt_label_path, cv2.IMREAD_GRAYSCALE)
                    label_img = label_img / 255
                    label_binary = cv2.resize(label_img, (self.train_input_size_w, self.train_input_size_h),
                                              interpolation=cv2.INTER_NEAREST)
                    label_binary = np.expand_dims(label_binary, axis=-1)
                    batch_labels_binary[num, :, :, :] = label_binary  # 0/1 矩阵 256x512x1

                    gt_label_path = self._gt_label_instance_list[index]
                    label_img = cv2.imread(gt_label_path, cv2.IMREAD_UNCHANGED)
                    label_instance = cv2.resize(label_img, (self.train_input_size_w, self.train_input_size_h),
                                           interpolation=cv2.INTER_NEAREST)
                    label_instance = np.expand_dims(label_instance, axis=-1)
                    batch_labels_instance[num, :, :, :] = label_instance  # 灰度图 256x512
                    num += 1
                self.batch_count += 1
                return batch_image, batch_labels_binary, batch_labels_instance
            else:
                self.batch_count = 0
                self._random_dataset()
                raise StopIteration


if __name__ == '__main__':
    val = DataSet('test')
    b1, b2, b3 = val.__next__()
    c1, c2, c3 = val.__next__()
    dd, d2, d3 = val.__next__()
