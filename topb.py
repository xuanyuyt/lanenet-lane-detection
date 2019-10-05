#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
from mylanenet_merge_model_work import LaneNet
import os
import pdb


pb_file = "D:\desktop\lane_work\lane_work\model\mobileNet_lanenet/culane_lanenet_mobilenet_v2_1005.pb"
#ckpt_file = cfg.YOLO.DEMO_WEIGHT
ckpt_file = 'D:\desktop\lane_work\lane_work\model\mobileNet_lanenet/culane_lanenet_mobilenet_v2_1005.ckpt'
# output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
#
#output_node_names = ["input_tensor", "lanenet_loss/inference/decode/score_final/Conv2D", "lanenet_loss/pix_embedding_conv/Conv2D"]
output_node_names = ["input_tensor", "lanenet_model/mobilenet_v2_frontend/decode/score_final/Conv2D","lanenet_model/mobilenet_v2_backend/instance_seg/pix_embedding_conv/Conv2D"]

input_data = tf.placeholder(dtype=tf.float32, name='input_tensor', shape=[None,None,None,3])
phase_tensor = tf.constant('false', tf.string)
net = LaneNet(phase=phase_tensor, net_flag='mobilenet_v2')
model=net.inference(input_data,name='lanenet_model')
#print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())