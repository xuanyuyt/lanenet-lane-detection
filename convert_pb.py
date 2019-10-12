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
from lanenet_model import lanenet


pb_file = "model/tusimple_lanenet_mobilenet_v2_1005/culane_lanenet_mobilenet_v2_1005.pb"
#ckpt_file = cfg.YOLO.DEMO_WEIGHT
ckpt_file = 'model/tusimple_lanenet_mobilenet_v2_1005/tusimple_lanenet_3600_0.929177263960692.ckpt-3601'
# output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
#
#output_node_names = ["input_tensor", "lanenet_loss/inference/decode/score_final/Conv2D", "lanenet_loss/pix_embedding_conv/Conv2D"]
output_node_names = ["input_tensor", "lanenet_model/mobilenet_v2_backend/binary_seg/ArgMax","lanenet_model/mobilenet_v2_backend/instance_seg/pix_embedding_conv/Conv2D"]

input_data = tf.placeholder(dtype=tf.float32, name='input_tensor', shape=[None,None,None,3])
phase_tensor = tf.constant('false', tf.string)
net = lanenet.LaneNet(phase=phase_tensor, net_flag='mobilenet_v2')
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