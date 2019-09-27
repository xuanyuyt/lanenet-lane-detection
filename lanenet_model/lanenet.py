#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午8:50
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet.py
# @IDE: PyCharm
"""
Implement LaneNet Model
"""
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNet(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, net_flag='vgg', reuse=False):
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._reuse = reuse
        if net_flag == 'mobilenet_v2':
            self._need_layer_norm = False
        else:
            self._need_layer_norm = True

        self._frontend = lanenet_front_end.LaneNetFrondEnd(
            phase=phase, net_flag=net_flag
        )
        self._backend = lanenet_back_end.LaneNetBackEnd(
            phase=phase
        )

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # # second apply backend process
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse,
                need_layer_norm = self._need_layer_norm
            )
            # pix_embedding_conv
            if not self._reuse:
                self._reuse = True

        return binary_seg_prediction, instance_seg_prediction

    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse,
                need_layer_norm=self._need_layer_norm
            )

            if not self._reuse:
                self._reuse = True

        return calculated_losses
    
    def compute_acc(self, input_tensor, binary_label_tensor, name):
        with tf.variable_scope(name_or_scope=name, reuse=self._reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=self._reuse
            )

            # # second apply backend process
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=self._reuse,
                need_layer_norm = self._need_layer_norm
            )
            # pix_embedding_conv
            if not self._reuse:
                self._reuse = True
                
        final_output = tf.expand_dims(binary_seg_prediction, axis=-1)
        idx = tf.where(tf.equal(final_output, 1))
        pix_cls_ret = tf.gather_nd(binary_label_tensor, idx)
        # =========================== recall to line
        recall = tf.count_nonzero(pix_cls_ret) # 车道线正检数
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
        precision = tf.subtract(tf.cast(tf.shape(pix_cls_ret)[0], tf.int64), tf.count_nonzero(pix_cls_ret)) # 背景正检数
        precision = tf.divide(precision, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))
        # =========================== accuracy
        accuracy = tf.divide(2.0, tf.divide(1.0, recall) + tf.divide(1.0, precision))
        return binary_seg_prediction, instance_seg_prediction, recall, fp, fn, precision, accuracy
        
