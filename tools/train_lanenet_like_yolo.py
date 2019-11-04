#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ================================================================
# @Time    : 2019/10/30 10:11
# @Author  : YangTao
# @Site    : 
# @File    : train_lanenet_like_yolo.py
# @IDE: PyCharm Community Edition
# ================================================================
"""

"""

import argparse
from tqdm import tqdm
import os.path as ops
import time
import glog as log
import numpy as np
import tensorflow as tf
import sys

sys.path.append('./')
import os

GPU_IDS = '7'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
from lanenet_model import lanenet
from tools import evaluate_model_utils
from config.global_config import cfg
from data_provider.lanenet_data_providr_like_yolov3 import DataSet


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', type=str,
                        # default='./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg_changename.ckpt',
                        default='./model/tusimple_lanenet_mobilenet_v2_1005/tusimple_lanenet_3600_0.929177263960692.ckpt-3601',
                        help='Path to pre-trained weights to continue training')

    parser.add_argument('-m', '--multi_gpus', type=bool, default=True,
                        help='Use multi gpus to train')
    parser.add_argument('--net_flag', type=str, default='mobilenet_v2',  # mobilenet_v2 vgg
                        help='The net flag which determins the net\'s architecture')
    parser.add_argument('--version_flag', type=str, default='1030',
                        help='The net flag which determins the net\'s architecture')
    parser.add_argument('--scratch', type=bool, default=True,
                        help='Is training from scratch ?')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def load_pretrained_weights(variables, pretrained_weights_path, sess):
    """

    :param variables:
    :param pretrained_weights_path:
    :param sess:
    :return:
    """
    assert ops.exists(pretrained_weights_path), '{:s} not exist'.format(pretrained_weights_path)

    pretrained_weights = np.load(
        './data/vgg16.npy', encoding='latin1').item()

    for vv in variables:
        weights_key = vv.name.split('/')[-3]
        if 'conv5' in weights_key:
            weights_key = '{:s}_{:s}'.format(weights_key.split('_')[0], weights_key.split('_')[1])
        try:
            weights = pretrained_weights[weights_key][0]
            _op = tf.assign(vv, weights)
            sess.run(_op)
        except Exception as _:
            continue

    return


def train_lanenet(weights_path=None, net_flag='vgg', version_flag='', scratch=False):
    """
    :param weights_path:
    :param net_flag: choose which base network to use
    :param version_flag: exp flag
    :return:
    """
    # ========================== placeholder ========================= #
    with tf.name_scope('train_input'):
        train_input_tensor = tf.placeholder(dtype=tf.float32, name='input_image',
                                      shape=[None, None, None, 3])
        train_binary_label_tensor = tf.placeholder(dtype=tf.float32, name='binary_input_label',
                                             shape=[None, None, None, 1])
        train_instance_label_tensor = tf.placeholder(dtype=tf.float32, name='instance_input_label',
                                               shape=[None, None, None,1])

    with tf.name_scope('val_input'):
        val_input_tensor = tf.placeholder(dtype=tf.float32, name='input_image',
                                      shape=[None, None, None, 3])
        val_binary_label_tensor = tf.placeholder(dtype=tf.float32, name='binary_input_label',
                                             shape=[None, None, None, 1])
        val_instance_label_tensor = tf.placeholder(dtype=tf.float32, name='instance_input_label',
                                               shape=[None, None, None,1])

    # ================================================================ #
    #                           Define Network                         #
    # ================================================================ #
    train_net = lanenet.LaneNet(net_flag=net_flag, phase='train', reuse=tf.AUTO_REUSE)
    val_net = lanenet.LaneNet(net_flag=net_flag, phase='val', reuse=True)
    # ---------------------------------------------------------------- #

    # ================================================================ #
    #                       Train Input & Output                       #
    # ================================================================ #
    trainset = DataSet('train')
    train_compute_ret = train_net.compute_loss(
        input_tensor=train_input_tensor, binary_label=train_binary_label_tensor,
        instance_label=train_instance_label_tensor, name='lanenet_model'
    )
    train_total_loss = train_compute_ret['total_loss']
    train_binary_seg_loss = train_compute_ret['binary_seg_loss']  # 语义分割 loss
    train_disc_loss = train_compute_ret['discriminative_loss']  # embedding loss
    train_pix_embedding = train_compute_ret['instance_seg_logits']  # embedding feature, HxWxN
    train_l2_reg_loss = train_compute_ret['l2_reg_loss']

    train_prediction_logits = train_compute_ret['binary_seg_logits']  # 语义分割结果，HxWx2
    train_prediction_score = tf.nn.softmax(logits=train_prediction_logits)
    train_prediction = tf.argmax(train_prediction_score, axis=-1)  # 语义分割二值图

    train_accuracy = evaluate_model_utils.calculate_model_precision(
        train_compute_ret['binary_seg_logits'], train_binary_label_tensor
    )
    train_fp = evaluate_model_utils.calculate_model_fp(
        train_compute_ret['binary_seg_logits'], train_binary_label_tensor
    )
    train_fn = evaluate_model_utils.calculate_model_fn(
        train_compute_ret['binary_seg_logits'], train_binary_label_tensor
    )
    train_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=train_prediction
    )  # (I - min) * 255 / (max -min), 归一化到0-255
    train_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=train_pix_embedding
    )  # (I - min) * 255 / (max -min), 归一化到0-255
    # ---------------------------------------------------------------- #

    # ================================================================ #
    #                          Define Optimizer                        #
    # ================================================================ #
    # set optimizer
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # learning_rate = tf.train.cosine_decay_restarts( # 余弦衰减
    #     learning_rate=cfg.TRAIN.LEARNING_RATE,      # 初始学习率
    #     global_step=global_step,                    # 当前迭代次数
    #     first_decay_steps=cfg.TRAIN.STEPS/3,        # 首次衰减周期
    #     t_mul=2.0,                                  # 随后每次衰减周期倍数
    #     m_mul=1.0,                                  # 随后每次初始学习率倍数
    #     alpha = 0.1,                                # 最小的学习率=alpha*learning_rate
    # )
    learning_rate = tf.train.polynomial_decay(  # 多项式衰减
        learning_rate=cfg.TRAIN.LEARNING_RATE,  # 初始学习率
        global_step=global_step,  # 当前迭代次数
        decay_steps=cfg.TRAIN.STEPS / 4,  # 在迭代到该次数实际，学习率衰减为 learning_rate * dacay_rate
        end_learning_rate=cfg.TRAIN.LEARNING_RATE / 10,  # 最小的学习率
        power=0.9,
        cycle=True
    )
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch normalization
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=cfg.TRAIN.MOMENTUM).minimize(
            loss=train_total_loss,
            var_list=tf.trainable_variables(),
            global_step=global_step
        )
    # ---------------------------------------------------------------- #

    # ================================================================ #
    #                           Train Summary                          #
    # ================================================================ #
    train_loss_scalar = tf.summary.scalar(
        name='train_cost', tensor=train_total_loss
    )
    train_accuracy_scalar = tf.summary.scalar(
        name='train_accuracy', tensor=train_accuracy
    )
    train_binary_seg_loss_scalar = tf.summary.scalar(
        name='train_binary_seg_loss', tensor=train_binary_seg_loss
    )
    train_instance_seg_loss_scalar = tf.summary.scalar(
        name='train_instance_seg_loss', tensor=train_disc_loss
    )
    train_fn_scalar = tf.summary.scalar(
        name='train_fn', tensor=train_fn
    )
    train_fp_scalar = tf.summary.scalar(
        name='train_fp', tensor=train_fp
    )
    train_binary_seg_ret_img = tf.summary.image(
        name='train_binary_seg_ret', tensor=train_binary_seg_ret_for_summary
    )
    train_embedding_feats_ret_img = tf.summary.image(
        name='train_embedding_feats_ret', tensor=train_embedding_ret_for_summary
    )
    train_merge_summary_op = tf.summary.merge(
        [train_accuracy_scalar, train_loss_scalar, train_binary_seg_loss_scalar,
         train_instance_seg_loss_scalar, train_fn_scalar, train_fp_scalar,
         train_binary_seg_ret_img, train_embedding_feats_ret_img,
         learning_rate_scalar]
    )
    # ---------------------------------------------------------------- #

    # ================================================================ #
    #                        Val Input & Output                        #
    # ================================================================ #
    valset = DataSet('val', net_flag)
    val_compute_ret = val_net.compute_loss(
        input_tensor=val_input_tensor, binary_label=val_binary_label_tensor,
        instance_label=val_instance_label_tensor, name='lanenet_model'
    )
    val_total_loss = val_compute_ret['total_loss']
    val_binary_seg_loss = val_compute_ret['binary_seg_loss']
    val_disc_loss = val_compute_ret['discriminative_loss']
    val_pix_embedding = val_compute_ret['instance_seg_logits']

    val_prediction_logits = val_compute_ret['binary_seg_logits']
    val_prediction_score = tf.nn.softmax(logits=val_prediction_logits)
    val_prediction = tf.argmax(val_prediction_score, axis=-1)

    val_accuracy = evaluate_model_utils.calculate_model_precision(
        val_compute_ret['binary_seg_logits'], val_binary_label_tensor
    )
    val_fp = evaluate_model_utils.calculate_model_fp(
        val_compute_ret['binary_seg_logits'], val_binary_label_tensor
    )
    val_fn = evaluate_model_utils.calculate_model_fn(
        val_compute_ret['binary_seg_logits'], val_binary_label_tensor
    )
    val_binary_seg_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=val_prediction
    )
    val_embedding_ret_for_summary = evaluate_model_utils.get_image_summary(
        img=val_pix_embedding
    )
    # ---------------------------------------------------------------- #

    # ================================================================ #
    #                            VAL Summary                           #
    # ================================================================ #
    val_loss_scalar = tf.summary.scalar(
        name='val_cost', tensor=val_total_loss
    )
    val_accuracy_scalar = tf.summary.scalar(
        name='val_accuracy', tensor=val_accuracy
    )
    val_binary_seg_loss_scalar = tf.summary.scalar(
        name='val_binary_seg_loss', tensor=val_binary_seg_loss
    )
    val_instance_seg_loss_scalar = tf.summary.scalar(
        name='val_instance_seg_loss', tensor=val_disc_loss
    )
    val_fn_scalar = tf.summary.scalar(
        name='val_fn', tensor=val_fn
    )
    val_fp_scalar = tf.summary.scalar(
        name='val_fp', tensor=val_fp
    )
    val_binary_seg_ret_img = tf.summary.image(
        name='val_binary_seg_ret', tensor=val_binary_seg_ret_for_summary
    )
    val_embedding_feats_ret_img = tf.summary.image(
        name='val_embedding_feats_ret', tensor=val_embedding_ret_for_summary
    )
    val_merge_summary_op = tf.summary.merge(
        [val_accuracy_scalar, val_loss_scalar, val_binary_seg_loss_scalar,
         val_instance_seg_loss_scalar, val_fn_scalar, val_fp_scalar,
         val_binary_seg_ret_img, val_embedding_feats_ret_img]
    )
    # ---------------------------------------------------------------- #

    # ================================================================ #
    #                      Config Saver & Session                      #
    # ================================================================ #
    # Set tf model save path
    model_save_dir = 'model/tusimple_lanenet_{:s}_{:s}'.format(net_flag, version_flag)
    os.makedirs(model_save_dir, exist_ok=True)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'tusimple_lanenet_{:s}_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # ==============================
    if scratch:
        """
        删除 Momentum 的参数, 注意这里保存的 meta 文件也会删了
        tensorflow 在 save model 的时候，如果选择了 global_step 选项，会 global_step 值也保存下来，
        然后 restore 的时候也就会接着这个 global_step 继续训练下去，因此需要去掉
        """
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_resotre = [v for v in variables if 'Momentum' not in v.name.split('/')[-1]]
        variables_to_resotre = [v for v in variables_to_resotre if
                                'global_step' not in v.name.split('/')[-1]]  # remove global step
        restore_saver = tf.train.Saver(variables_to_resotre)
    else:
        restore_saver = tf.train.Saver()
    saver = tf.train.Saver(max_to_keep=10)
    # ==============================

    # Set tf summary save path
    tboard_save_path = 'tboard/tusimple_lanenet_{:s}_{:s}'.format(net_flag, version_flag)
    os.makedirs(tboard_save_path, exist_ok=True)

    # Set sess configuration
    # ============================== config GPU
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    # ==============================
    sess = tf.Session(config=sess_config)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)
    # ---------------------------------------------------------------- #

    # Set the training parameters
    import math
    one_epoch2step = math.ceil(cfg.TRAIN.TRAIN_SIZE / cfg.TRAIN.BATCH_SIZE)  # 训练一个 epoch 需要的 batch 数量
    total_epoch = math.ceil(cfg.TRAIN.STEPS / one_epoch2step) # 一共需要训练多少 epoch

    log.info('Global configuration is as follows:')
    log.info(cfg)
    max_acc = 0.9
    save_num = 0
    val_step = 0
    # ================================================================ #
    #                            Train & Val                           #
    # ================================================================ #
    with sess.as_default():
        # ============================== load pretrain model
        if weights_path is None:
            log.info('Training from scratch')
            sess.run(tf.global_variables_initializer())
        elif net_flag == 'vgg' and weights_path is None:
            load_pretrained_weights(tf.trainable_variables(), './data/vgg16.npy', sess)
        elif scratch: # 从头开始训练，类似 Caffe 的 --weights
            sess.run(tf.global_variables_initializer())
            log.info('Restore model from last model checkpoint {:s}, scratch'.format(weights_path))
            try:
                restore_saver.restore(sess=sess, save_path=weights_path)
            except:
                log.info('model maybe is not exist!')
        else: # 继续训练，类似 Caffe 的 --snapshot
            log.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            try:
                restore_saver.restore(sess=sess, save_path=weights_path)
            except:
                log.info('model maybe is not exist!')
        # ==============================
        for epoch in range(total_epoch):
            # ================================================================ #
            #                               Train                              #
            # ================================================================ #
            train_epoch_loss = []
            pbar_train = tqdm(trainset)
            train_t_start = time.time()
            for gt_imgs,  binary_gt_labels, instance_gt_labels in pbar_train:
                _, global_step_val, train_loss, train_accuracy_figure, train_fn_figure, train_fp_figure, \
                lr, train_summary, train_binary_loss, train_instance_loss, \
                train_embeddings, train_binary_seg_imgs, train_l2_loss = \
                    sess.run([optimizer, global_step, train_total_loss, train_accuracy, train_fn, train_fp,
                              learning_rate, train_merge_summary_op, train_binary_seg_loss,
                              train_disc_loss, train_pix_embedding, train_prediction, train_l2_reg_loss],
                             feed_dict={train_input_tensor: gt_imgs,
                                        train_binary_label_tensor: binary_gt_labels,
                                        train_instance_label_tensor: instance_gt_labels}
                             )
                # ============================== 透心凉，心飞扬
                if math.isnan(train_loss) or math.isnan(train_binary_loss) or math.isnan(train_instance_loss):
                    log.error('cost is: {:.5f}'.format(train_loss))
                    log.error('binary cost is: {:.5f}'.format(train_binary_loss))
                    log.error('instance cost is: {:.5f}'.format(train_instance_loss))
                    return
                # ==============================
                train_epoch_loss.append(train_loss)
                summary_writer.add_summary(summary=train_summary, global_step=global_step_val)
                pbar_train.set_description(("train loss: %.4f, learn rate: %e") % (train_loss, lr))
            train_cost_time = time.time() - train_t_start
            mean_train_loss = np.mean(train_epoch_loss)
            log.info('MEAN Train: total_loss= {:6f} mean_cost_time= {:5f}s'
                     .format(mean_train_loss, train_cost_time))
            # ---------------------------------------------------------------- #

            # ================================================================ #
            #                                Val                               #
            # ================================================================ #
            # 每隔 epoch 次，测试整个验证集
            pbar_val = tqdm(valset)
            val_epoch_loss = []
            val_epoch_binary_loss = []
            val_epoch_instance_loss = []
            val_epoch_accuracy_figure = []
            val_epoch_fp_figure = []
            val_epoch_fn_figure = []
            val_t_start = time.time()
            for val_images, val_binary_labels, val_instance_labels in pbar_val:
                # validation part
                val_step += 1
                val_summary, \
                val_loss, val_binary_loss, val_instance_loss, \
                val_accuracy_figure, val_fn_figure, val_fp_figure = \
                    sess.run([val_merge_summary_op,
                              val_total_loss, val_binary_seg_loss, val_disc_loss,
                              val_accuracy, val_fn, val_fp],
                             feed_dict={val_input_tensor: val_images,
                                        val_binary_label_tensor: val_binary_labels,
                                        val_instance_label_tensor: val_instance_labels}
                             )
                # ============================== 透心凉，心飞扬
                if math.isnan(val_loss) or math.isnan(val_binary_loss) or math.isnan(val_instance_loss):
                    log.error('cost is: {:.5f}'.format(val_loss))
                    log.error('binary cost is: {:.5f}'.format(val_binary_loss))
                    log.error('instance cost is: {:.5f}'.format(val_instance_loss))
                    return
                # ==============================
                summary_writer.add_summary(summary=val_summary, global_step=val_step)
                pbar_val.set_description(("val loss: %.4f, accuracy: %.4f") % (train_loss, val_accuracy_figure))

                val_epoch_loss.append(val_loss)
                val_epoch_binary_loss.append(val_binary_loss)
                val_epoch_instance_loss.append(val_instance_loss)
                val_epoch_accuracy_figure.append(val_accuracy_figure)
                val_epoch_fp_figure.append(val_fp_figure)
                val_epoch_fn_figure.append(val_fn_figure)
            val_cost_time = time.time() - val_t_start
            mean_val_loss = np.mean(val_epoch_loss)
            mean_val_binary_loss = np.mean(val_epoch_binary_loss)
            mean_val_instance_loss = np.mean(val_epoch_instance_loss)
            mean_val_accuracy_figure = np.mean(val_epoch_accuracy_figure)
            mean_val_fp_figure = np.mean(val_epoch_fp_figure)
            mean_val_fn_figure = np.mean(val_epoch_fn_figure)

            # ==============================
            if mean_val_accuracy_figure > max_acc:
                max_acc = mean_val_accuracy_figure
                if save_num < 3:  # 前三次不算
                    max_acc = 0.9
                log.info('MAX_ACC change to {}'.format(mean_val_accuracy_figure))
                model_save_path_max = ops.join(model_save_dir,
                                               'tusimple_lanenet_{}.ckpt'.format(mean_val_accuracy_figure))
                saver.save(sess=sess, save_path=model_save_path_max, global_step=global_step)
                save_num += 1
            # ==============================

            log.info('=> Epoch: {}, MEAN Val: total_loss= {:6f} binary_seg_loss= {:6f} '
                     'instance_seg_loss= {:6f} accuracy= {:6f} fp= {:6f} fn= {:6f}'
                     ' mean_cost_time= {:5f}s '.
                     format(epoch, mean_val_loss, mean_val_binary_loss, mean_val_instance_loss,
                            mean_val_accuracy_figure,mean_val_fp_figure, mean_val_fn_figure, val_cost_time))
            # ---------------------------------------------------------------- #
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if len(GPU_IDS.split(',')) < 2:
        args.multi_gpus = False
    print('GPU_IDS: ', GPU_IDS)
    # train lanenet
    if not args.multi_gpus:
        train_lanenet(args.weights_path, net_flag=args.net_flag,
                      version_flag=args.version_flag, scratch=args.scratch)
    else:
        pass
