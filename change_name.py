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
import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--train_from_coco", action='store_true')
# flag = parser.parse_args()

org_weights_path = "./model/mobileNet_lanenet/culane_lanenet_mobilenet_v2_2018-09-09-20-18-15.ckpt"
cur_weights_path = "./model/mobileNet_lanenet/culane_lanenet_mobilenet_v2.ckpt"

def rename_var(ckpt_path, new_ckpt_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            print(var_name)
            var = tf.contrib.framework.load_variable(ckpt_path, var_name)
            var_name = var_name.replace('lanenet_loss', 'lanenet_model')
            var_name = var_name.replace('W', 'w')
            var_name = var_name.replace('inference/encode', 'mobilenet_v2_frontend/encode')
            var_name = var_name.replace('inference/decode', 'mobilenet_v2_frontend/decode')
            var_name = var_name.replace('pix_embedding_conv', 'mobilenet_v2_backend/instance_seg/pix_embedding_conv')
            var_name = var_name.replace('pix_embedding_conv', 'mobilenet_v2_backend/instance_seg/pix_embedding_conv')
            var = tf.Variable(var, name=var_name)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, new_ckpt_path)

rename_var(org_weights_path, cur_weights_path)



#
#
#
#
#
#
#
#
#
# preserve_cur_names = ['lanenet_model']
# preserve_org_names = ['lanenet_loss']
#
# org_weights_mess = []
# tf.Graph().as_default()
# load = tf.train.import_meta_graph(org_weights_path + '.meta')
# with tf.Session() as sess:
#     load.restore(sess, org_weights_path)
#     for var in tf.global_variables():
#         var_name = var.op.name
#         var_name_mess = str(var_name).split('/')
#         var_shape = var.shape
#         if True:
#             if (var_name_mess[-1] in ['beta1_power', 'beta2_power', 'Adam', 'Adam_1'] or var_name_mess[0] in [
#                 'learn_rate', 'Variable']):
#                 continue
#         # with open('/data/panting/tensorflow-yolov3/meta_weights_mess.txt', 'a')as f:
#         #     f.write(str([var_name, var_shape]))
#         #     f.write('\n')
#         org_weights_mess.append([var_name, var_shape])
# #         print("=> " + str(var_name).ljust(50), var_shape)
# # print()
# tf.reset_default_graph()
#
# cur_weights_mess = []
# tf.Graph().as_default()
# input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
# phase_tensor = tf.constant('train', tf.string)
# net = LaneNet(phase=phase_tensor, net_flag='mobilenet_v2')
# model=net.inference(input_tensor,name='lanenet_loss')
# for var in tf.global_variables():
#     var_name = var.op.name
#     var_name_mess = str(var_name).split('/')
#     var_shape = var.shape
#     print(var_name_mess[0])
#     if True:
#         if var_name_mess[0] in preserve_cur_names: continue
#     # with open('/data/panting/tensorflow-yolov3/yolov3_weights_mess.txt', 'a')as f:
#     #     f.write(str([var_name, var_shape]) + '\n')
#     cur_weights_mess.append([var_name, var_shape])
# #     print("=> " + str(var_name).ljust(50), var_shape)
# # with open('/data/panting/tensorflow-yolov3/org_weights_mess.txt','w')as f:
# #     f.write(str(org_weights_mess))
# # with open('/data/panting/tensorflow-yolov3/cur_weights_mess.txt','w')as f:
# #     f.write(str(cur_weights_mess))
# org_weights_num = len(org_weights_mess)
# cur_weights_num = len(cur_weights_mess)
# if cur_weights_num != org_weights_num:
#     raise RuntimeError
#
# print('=> Number of weights that will rename:\t%d' % cur_weights_num)
# cur_to_org_dict = {}
# for index in range(org_weights_num):
#     org_name, org_shape = org_weights_mess[index]
#     cur_name, cur_shape = cur_weights_mess[index]
#     if cur_shape != org_shape:
#         print(org_weights_mess[index])
#         print(cur_weights_mess[index])
#         raise RuntimeError
#     cur_to_org_dict[cur_name] = org_name
#     print("=> " + str(cur_name).ljust(50) + ' : ' + org_name)
#
# with tf.name_scope('load_save'):
#     # with tf.name_scope('loader_and_saver'):
#     #     self.loader = tf.train.Saver(self.net_var) #self.net_var
#     #     self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=10)
#     name_to_var_dict = {var.op.name: var for var in tf.global_variables()}
#     restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
#     load = tf.train.Saver(restore_dict)
#     save = tf.train.Saver(tf.global_variables())
#     for var in tf.global_variables():
#         print("=> " + var.op.name)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print('=> Restoring weights from:\t %s' % org_weights_path)
#     load.restore(sess, org_weights_path)
#     save.save(sess, cur_weights_path)
# tf.reset_default_graph()