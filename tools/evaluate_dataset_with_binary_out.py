#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
# @Time    : 2019/12/13 11:22
# @Author  : YangTao
# @Site    : 
# @File    : evaluate_dataset_with_binary_out.py
# @IDE: PyCharm Community Edition
# ================================================================
import os.path as ops
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def compute_acc(binary_label_tensor, binary_seg_prediction):
    final_output = tf.expand_dims(binary_seg_prediction, axis=-1)
    idx = tf.where(tf.equal(final_output, 1))
    pix_cls_ret = tf.gather_nd(binary_label_tensor, idx)
    # =========================== recall to line
    recall = tf.count_nonzero(pix_cls_ret)  # 车道线正检数
    recall = tf.divide(
        recall,
        tf.cast(tf.shape(tf.gather_nd(binary_label_tensor, tf.where(tf.equal(binary_label_tensor, 1))))[0], tf.int64))
    # =========================== fp 车道线误检
    false_pred = tf.cast(tf.shape(pix_cls_ret)[0], tf.int64) - tf.count_nonzero(
        tf.gather_nd(binary_label_tensor, idx)
    )
    fp = tf.divide(false_pred, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))
    # =========================== fn 车道线漏检
    label_cls_ret = tf.gather_nd(binary_label_tensor, tf.where(tf.equal(binary_label_tensor, 1)))
    mis_pred = tf.cast(tf.shape(label_cls_ret)[0], tf.int64) - tf.count_nonzero(pix_cls_ret)
    fn = tf.divide(mis_pred, tf.cast(tf.shape(label_cls_ret)[0], tf.int64))
    # =========================== precision to background
    idx = tf.where(tf.equal(binary_label_tensor, 0))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    precision = tf.subtract(tf.cast(tf.shape(pix_cls_ret)[0], tf.int64), tf.count_nonzero(pix_cls_ret))  # 背景正检数
    precision = tf.divide(precision, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))
    # =========================== accuracy
    accuracy = tf.divide(2.0, tf.divide(1.0, recall) + tf.divide(1.0, precision))

    return recall, fp, fn, precision, accuracy

image_list = 'H:/Other_DataSets/TuSimple/val.txt'
gt_label_binary_list = []
predict_binary_list = []
input_dir =  ops.dirname(ops.abspath(image_list))
out_dir = 'H:/Other_DataSets/TuSimple/out/'
with open(image_list, 'r') as file:
    for _info in file:
        info_tmp = _info.strip(' ').split()
        gt_label_binary_list.append(ops.join(input_dir, info_tmp[1]))
        predict_binary_list.append(ops.join(out_dir, info_tmp[1]))
binary_label = tf.placeholder(dtype=tf.int64, shape=[None, 256, 512, 1], name='binary_input_label')
binary_prediction = tf.placeholder(dtype=tf.int64, shape=[None, 256, 512], name='binary_seg_prediction')
recall, fp, fn, precision, accuracy = compute_acc(binary_label, binary_prediction)
sess = tf.Session()
mean_accuracy = 0.0
mean_recall = 0.0
mean_precision = 0.0
mean_fp = 0.0
mean_fn = 0.0
total_num = 0
with sess.as_default():
    t_start = time.time()
    for gt_path, pre_img in zip(gt_label_binary_list, predict_binary_list):
        label_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        label_img = label_img / 255
        label_binary = cv2.resize(label_img, (512, 256),
                                  interpolation=cv2.INTER_NEAREST)
        plt.imshow(label_binary)
        plt.show()
        label_binary = np.expand_dims(label_binary, axis=-1)
        label_binary = np.expand_dims(label_binary, axis=0)
        pre_img = cv2.imread(pre_img, cv2.IMREAD_GRAYSCALE)
        pre_img = pre_img / 255
        plt.imshow(pre_img)
        plt.show()
        pre_img = np.expand_dims(pre_img, axis=0)
        recall_, fp_, fn_, precision_, accuracy_ = sess.run([recall, fp, fn, precision, accuracy],
                                                            feed_dict={binary_label: label_binary,
                                                                       binary_prediction: pre_img})
        print(recall_, fp_, fn_)
        mean_accuracy += accuracy
        mean_precision += precision
        mean_recall += recall
        mean_fp += fp
        mean_fn += fn
        total_num += 1
    t_cost = time.time() - t_start
    mean_accuracy = mean_accuracy / total_num
    mean_precision = mean_precision / total_num
    mean_recall = mean_recall / total_num
    mean_fp = mean_fp / total_num
    mean_fn = mean_fn / total_num
    print('测试 {} 张图片，耗时{}，{}_recall = {}, precision = {}, accuracy = {}, fp = {}, fn = {}, '.format(
        total_num, t_cost, 'mobilenet_v2', mean_recall, mean_precision, mean_accuracy, mean_fp, mean_fn))



