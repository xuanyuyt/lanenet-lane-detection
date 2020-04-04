#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
# @Time    : 2020/4/1 17:55
# @Author  : YangTao
# @Func    : 
# @File    : show_ori_data.py
# @IDE: PyCharm Community Edition
# ================================================================
import cv2
import json
import numpy as np
import os.path as ops
import argparse
from data_provider import lanenet_data_feed_pipline
import tensorflow as tf

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dataset_dir', type=str, default='H:/Other_DataSets/TuSimple/test_set/',
                        help='The source nsfw data dir path')
    parser.add_argument('--json_file_name', type=str, default='test_label.json',
                        help='The json annotation file name')
    parser.add_argument('--tf_record_dir', type=str, default='D:/Other_DataSets/TuSimple/',
                        help='The tf_record file dir')

    return parser.parse_args()


def show_ori_image(base_path, json_file_path):
    json_file_path = ops.join(base_path, json_file_path)
    image_num = 0
    max_lines = 0

    with open(json_file_path, 'r') as file:
        for line_index, line in enumerate(file):
            info_dict = json.loads(line)
            image = cv2.imread(base_path + info_dict['raw_file'])
            binaryimage = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
            instanceimage = binaryimage.copy()
            lanes = info_dict['lanes']
            h_samples = info_dict['h_samples']
            usefull_lane = 0

            for lane_index, lane in enumerate(lanes):
                assert len(h_samples) == len(lane)
                lane_x = []
                lane_y = []
                for index in range(len(lane)):
                    if lane[index] == -2:
                        continue
                    else:
                        ptx = lane[index]
                        pty = h_samples[index]
                        lane_x.append(ptx)
                        lane_y.append(pty)
                if not lane_x:
                    labeled = False
                    continue
                usefull_lane += 1
                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)

                cv2.polylines(binaryimage, lane_pts, isClosed=False,
                              color=255, thickness=5)
                cv2.polylines(instanceimage, lane_pts, isClosed=False,
                              color=lane_index * 50 + 20, thickness=5)

            if usefull_lane > max_lines:
                max_lines = usefull_lane

            cv2.imshow('image.jpg', image)
            cv2.imshow('binaryimage.jpg', binaryimage)
            cv2.imshow('instanceimage.jpg', instanceimage)
            cv2.waitKey(0)
            image_num = image_num + 1

    print("total {} image".format(image_num))
    print("max {} lines".format(max_lines))


def show_tf_record(tf_record_dir):
    train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=tf_record_dir, flags='train'
    )
    val_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(
        dataset_dir=tf_record_dir, flags='val'
    )

    # set compute graph node for training
    train_images, train_binary_labels, train_instance_labels = train_dataset.inputs(10)

    # set compute graph node for validation
    val_images, val_binary_labels, val_instance_labels = val_dataset.inputs(10)

    # Set sess configuration
    # ============================== config GPU
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.allocator_type = 'BFC'
    # ==============================
    sess = tf.Session(config=sess_config)

    with sess.as_default():
        train_images_np, train_binary_labels_np, train_instance_labels_np = sess.run(
            [train_images, train_binary_labels, train_instance_labels]
        )
        print()


if __name__ == '__main__':
    # init args
    args = init_args()

    # assert ops.exists(args.dataset_dir), '{:s} not exist'.format(args.dataset_dir)
    # show_ori_image(args.image_dataset_dir, args.json_file_name)

    show_tf_record(args.tf_record_dir)




