#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 14:19
# @Author  : YangTao
# @Site    : 
# @File    : evaluate_dataset.py
# @IDE: PyCharm Community Edition
"""
use model to evalueate Tusimple dataset
"""

import os
import os.path as ops
import argparse
import time
import math
import sys
sys.path.append('./')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import glog as log
import numpy as np
import cv2

from lanenet_model import lanenet
from data_provider import lanenet_data_processor
from config import global_config

CFG = global_config.cfg


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str,
                        default = 'H:/Other_DataSets/TuSimple/val.txt',
                        # default='../../TuSimple/val.txt',
                        help='The image list path')
    parser.add_argument('--weights_path', type=str,
                        default='./model/tusimple_lanenet_mobilenet_v2_1005/tusimple_lanenet_3600_0.929177263960692.ckpt-3601',
                        # default='./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg_changename.ckpt',
                        help='The lanenet model weights path')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=1)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=0)
    parser.add_argument('--net_flag', type=str, default='mobilenet_v2', # vgg mobilenet_v2
                        help='Backbone Network Tag')

    return parser.parse_args()


def test_lanenet_batch(image_list, weights_path, batch_size, use_gpu, net_flag='vgg'):
    """

    :param image_list:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param net_flag:
    :return:
    """
    assert ops.exists(image_list), '{:s} not exist'.format(image_list)

    log.info('开始加载数据集列表...')
    test_dataset = lanenet_data_processor.DataSet(image_list, traing=False)

    # ==============================
    gt_label_binary_list = []
    with open(image_list, 'r') as file:
        for _info in file:
            info_tmp = _info.strip(' ').split()
            gt_label_binary_list.append(info_tmp[1])
    # ==============================

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                         shape=[None, 256, 512, 1], name='binary_input_label')
    phase_tensor = tf.constant('test', tf.string)
    net = lanenet.LaneNet(phase=phase_tensor, net_flag=net_flag)
    binary_seg_ret, instance_seg_ret, recall_ret, false_positive, false_negative, precision_ret, accuracy_ret = \
        net.compute_acc(input_tensor=input_tensor, binary_label_tensor=binary_label_tensor, name='lanenet_model')

    saver = tf.train.Saver()
    # ==============================
    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    # ==============================
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        epoch_nums = int(math.ceil(test_dataset._dataset_size / batch_size))
        mean_accuracy = 0.0
        mean_recall = 0.0
        mean_precision = 0.0
        mean_fp = 0.0
        mean_fn = 0.0
        total_num = 0
        t_start = time.time()
        for epoch in range(epoch_nums):
            gt_imgs, binary_gt_labels, instance_gt_labels = test_dataset.next_batch(batch_size)
            if net_flag == 'vgg':
                image_list_epoch = [tmp / 127.5 - 1.0 for tmp in gt_imgs]
            elif net_flag == 'mobilenet_v2':
                image_list_epoch = [tmp - [103.939, 116.779, 123.68] for tmp in gt_imgs]

            binary_seg_images, instance_seg_images, recall, fp, fn, precision, accuracy = sess.run(
                [binary_seg_ret, instance_seg_ret, recall_ret, false_positive, false_negative, precision_ret, accuracy_ret],
                feed_dict={input_tensor: image_list_epoch, binary_label_tensor: binary_gt_labels})
            # ==============================
            out_dir = 'H:/Other_DataSets/TuSimple/out/'
            dst_binary_image_path = ops.join(out_dir,gt_label_binary_list[epoch])
            root_dir = ops.dirname(ops.abspath(dst_binary_image_path))
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            cv2.imwrite(dst_binary_image_path, binary_seg_images[0] * 255)
            # ==============================
            print(recall, fp, fn)
            mean_accuracy += accuracy
            mean_precision += precision
            mean_recall += recall
            mean_fp += fp
            mean_fn += fn
            total_num += len(gt_imgs)
        t_cost = time.time() - t_start
        mean_accuracy = mean_accuracy / epoch_nums
        mean_precision = mean_precision / epoch_nums
        mean_recall = mean_recall / epoch_nums
        mean_fp = mean_fp / epoch_nums
        mean_fn = mean_fn / epoch_nums
        print('测试 {} 张图片，耗时{}，{}_recall = {}, precision = {}, accuracy = {}, fp = {}, fn = {}, '.format(
            total_num, t_cost, net_flag, mean_recall, mean_precision, mean_accuracy, mean_fp, mean_fn))

    sess.close()

"""
测试 2782 张图片，耗时529.4126558303833，mobilenet_v2_recall = 0.9335565098483821, precision = 0.9500883211213997, 
accuracy = 0.9400811002186817, fp = 0.8713319645920296, fn = 0.06644349015161731, 
"""


if __name__ == '__main__':
    # init args
    args = init_args()

    test_lanenet_batch(image_list=args.image_list, weights_path=args.weights_path,
                       use_gpu=args.use_gpu, batch_size=args.batch_size, net_flag=args.net_flag)



