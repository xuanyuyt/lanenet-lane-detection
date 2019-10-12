#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_weight.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:51:31
#   Description :
#
# ================================================================

import argparse
import tensorflow as tf
from lanenet_model import lanenet
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# parser = argparse.ArgumentParser()
# parser.add_argument("--train_from_coco", action='store_true')
# flag = parser.parse_args()

org_weights_path = "model/tusimple_lanenet_mobilenet_v2_1005/tusimple_lanenet_3600_0.929177263960692.ckpt-3601"
cur_weights_path = "model/tusimple_lanenet_mobilenet_v2_1005/culane_lanenet_mobilenet_v2_1005_reduce_train.ckpt"

org_weights_mess = []
tf.Graph().as_default()
load = tf.train.import_meta_graph(org_weights_path + '.meta')
with tf.Session() as sess:
    load.restore(sess, org_weights_path)
    for var in tf.global_variables():
        var_name = var.op.name
        var_name_mess = str(var_name).split('/')
        var_shape = var.shape
        if True:
            if (var_name_mess[-1] in ['Momentum'] or var_name_mess[0] in ['learn_rate','Variable']):
                continue

        with open('model/tusimple_lanenet_mobilenet_v2_1005/a_meta_weights_mess.txt', 'a')as f:
            f.write(str([var_name, var_shape]))
            f.write('\n')
        org_weights_mess.append([var_name, var_shape])
tf.reset_default_graph()
# pdb.set_trace()

cur_weights_mess = []
tf.Graph().as_default()
input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
phase_tensor = tf.constant('train', tf.string)
net = lanenet.LaneNet(phase=phase_tensor, net_flag='mobilenet_v2')
model=net.inference(input_tensor,name='lanenet_model')
for var in tf.global_variables():
    var_name = var.op.name
    var_name_mess = str(var_name).split('/')
    var_shape = var.shape
    with open('model/tusimple_lanenet_mobilenet_v2_1005/a_meta_weights_mess.txt', 'a')as f:
        f.write(str([var_name, var_shape]) + '\n')
    cur_weights_mess.append([var_name, var_shape])

# pdb.set_trace()
org_weights_num = len(org_weights_mess)
cur_weights_num = len(cur_weights_mess)
# pdb.set_trace()
if cur_weights_num != org_weights_num:
    raise RuntimeError

print('=> Number of weights that will rename:\t%d' % cur_weights_num)
cur_to_org_dict = {}
for index in range(org_weights_num):
    org_name, org_shape = org_weights_mess[index]
    cur_name, cur_shape = cur_weights_mess[index]
    if cur_shape != org_shape:
        print(org_weights_mess[index])
        print(cur_weights_mess[index])
        raise RuntimeError
    cur_to_org_dict[cur_name] = org_name
    print("=> " + str(cur_name).ljust(50) + ' : ' + org_name)

with tf.name_scope('load_save'):
    name_to_var_dict = {var.op.name: var for var in tf.global_variables()}
    restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
    load = tf.train.Saver(restore_dict)
    save = tf.train.Saver(tf.global_variables())
    for var in tf.global_variables():
        print("=> " + var.op.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('=> Restoring weights from:\t %s' % org_weights_path)
    load.restore(sess, org_weights_path)
    save.save(sess, cur_weights_path)
tf.reset_default_graph()

