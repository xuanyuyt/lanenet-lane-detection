#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午6:42
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : mobilev2_based_fcn.py
# @IDE: PyCharm
"""
Implement mobilev2 based fcn net for semantic segmentation
"""
import collections

import tensorflow as tf

from config import global_config
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class MOBILEV2FCN(cnn_basenet.CNNBaseModel):
    """
    VGG 16 based fcn net for semantic segmentation
    """
    def __init__(self, phase):
        """

        """
        super(MOBILEV2FCN, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._net_intermediate_results = collections.OrderedDict()

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)
        return tf.equal(phase, tf.constant('train', dtype=tf.string))

    def _mobilev2_conv_stage(self, input_tensor, k_size, out_dims, name,
                          stride=1, pad='SAME', need_layer_norm=True):
        """
        stack conv and activation in mobilev2
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param pad:
        :param need_layer_norm:
        :return:
        """
        with tf.name_scope(name), tf.variable_scope(name):
            conv = self.conv2d(
                inputdata=input_tensor, out_channel=out_dims,
                kernel_size=k_size, stride=stride,
                use_bias=False, padding=pad, name='conv2d'
            )

            if need_layer_norm:
                bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')

                relu = self.relu(inputdata=bn, name='relu')
            else:
                relu = self.relu(inputdata=conv, name='relu')

        return relu

    def _res_block(self, input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
        with tf.name_scope(name), tf.variable_scope(name):
            # pw
            bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
            with tf.name_scope('pw'):
                net = self.conv2d(inputdata=input, out_channel=bottleneck_dim,
                                  kernel_size=1, name='pw', use_bias=bias)
            
            net = self.layerbn(net, is_training=is_train, name='pw_bn')
            net = self.relu6(net)
            # dw
            net = self.dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
            net = self.layerbn(net, is_training=is_train, name='dw_bn')
            net = self.relu6(net)
            # pw & linear
            with tf.name_scope('pw_linear'):
                net = self.conv2d(inputdata=net, out_channel=output_dim,
                                  kernel_size=1, name='pw_linear', use_bias=bias)
            net = self.layerbn(net, is_training=is_train, name='pw_linear_bn')

            # element wise add, only for stride==1
            if shortcut and stride == 1:
                in_dim = int(input.get_shape().as_list()[-1])
                if in_dim != output_dim:
                    with tf.name_scope('ex_dim'):
                        ins = self.conv2d(inputdata=input, out_channel=output_dim,
                                          kernel_size=1, name='ex_dim', use_bias=bias)
                    net = ins + net
                else:
                    net = input + net

            return net

    def _decode_block(self, input_tensor, previous_feats_tensor,
                      out_channels_nums, name, kernel_size=4,
                      stride=2, use_bias=False,
                      previous_kernel_size=4, need_activate=True):
        """

        :param input_tensor:
        :param previous_feats_tensor:
        :param out_channels_nums:
        :param kernel_size:
        :param previous_kernel_size:
        :param use_bias:
        :param stride:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):

            deconv_weights_stddev = tf.sqrt(
                tf.divide(tf.constant(2.0, tf.float32),
                          tf.multiply(tf.cast(previous_kernel_size * previous_kernel_size, tf.float32),
                                      tf.cast(tf.shape(input_tensor)[3], tf.float32)))
            ) # sqrt( 2 / (previous_kernel_size * previous_kernel_size * input_tensor.shape[3)
            deconv_weights_init = tf.truncated_normal_initializer(
                mean=0.0, stddev=deconv_weights_stddev)

            deconv = self.deconv2d(
                inputdata=input_tensor, out_channel=out_channels_nums, kernel_size=kernel_size,
                stride=stride, use_bias=use_bias, w_init=deconv_weights_init,
                name='deconv'
            )

            deconv = self.layerbn(inputdata=deconv, is_training=self._is_training, name='deconv_bn')

            deconv = self.relu(inputdata=deconv, name='deconv_relu')

            fuse_feats = tf.add(
                previous_feats_tensor, deconv, name='fuse_feats'
            )

            if need_activate:

                fuse_feats = self.layerbn(
                    inputdata=fuse_feats, is_training=self._is_training, name='fuse_gn'
                )

                fuse_feats = self.relu(inputdata=fuse_feats, name='fuse_relu')

        return fuse_feats

    def _mobilev2_fcn_encode(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            exp = 6

            # encode stage 1
            conv_1_1 = self._mobilev2_conv_stage(
                input_tensor=input_tensor, k_size=3,
                out_dims=32, name='conv1_1',stride=2,
                need_layer_norm=True
            )

            # encode stage 2
            res2_1 = self._res_block(conv_1_1, 1, 16, 1, self._is_training, name='res2_1')
            res3_1 = self._res_block(res2_1, exp, 24, 2, self._is_training, name='res3_1')  # size/4

            # encode stage 3
            res4_1 = self._res_block(res3_1, exp, 32, 2, self._is_training, name='res4_1')  # size/8
            res4_2 = self._res_block(res4_1, exp, 32, 1, self._is_training, name='res4_2')
            res4_3 = self._res_block(res4_2, exp, 32, 1, self._is_training, name='res4_3')

            res5_1 = self._res_block(res4_3, exp, 64, 1, self._is_training, name='res5_1')
            res5_2 = self._res_block(res5_1, exp, 64, 1, self._is_training, name='res5_2')
            res5_3 = self._res_block(res5_2, exp, 64, 1, self._is_training, name='res5_3')
            res5_4 = self._res_block(res5_3, exp, 64, 1, self._is_training, name='res5_4')

            self._net_intermediate_results['res5_4'] = {
                'data': res5_4,
                'shape': res5_4.get_shape().as_list()
            }

            # encode stage 4
            res6_1 = self._res_block(res5_4, exp, 96, 2, self._is_training, name='res6_1')  # size/16
            res6_2 = self._res_block(res6_1, exp, 96, 1, self._is_training, name='res6_2')
            res6_3 = self._res_block(res6_2, exp, 96, 1, self._is_training, name='res6_3')

            self._net_intermediate_results['res6_3'] = {
                'data': res6_3,
                'shape': res6_3.get_shape().as_list()
            }

            # encode stage 5 for segmentation
            res7_1 = self._res_block(res6_3, exp, 160, 2, self._is_training, name='res7_1')  # size/32
            res7_2 = self._res_block(res7_1, exp, 160, 1, self._is_training, name='res7_2')
            res7_3 = self._res_block(res7_2, exp, 160, 1, self._is_training, name='res7_3')

            self._net_intermediate_results['res7_3'] = {
                'data': res7_3,
                'shape': res7_3.get_shape().as_list()
            }

        return

    def _mobilev2_fcn_decode(self, name):
        """

        :return:
        """
        decode_layer_list = ['res7_3',
                             'res6_3',
                             'res5_4']

        # decode part for binary segmentation
        with tf.variable_scope(name):
            # score stage 1
            input_tensor = self._net_intermediate_results[decode_layer_list[0]]['data']

            score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                kernel_size=1, use_bias=False, name='score_origin')
            decode_layer_list = decode_layer_list[1:]
            for i in range(len(decode_layer_list)):
                deconv = self.deconv2d(inputdata=score, out_channel=64, kernel_size=4,
                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))
                input_tensor = self._net_intermediate_results[decode_layer_list[i]]['data']
                score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))
                fused = tf.add(deconv, score, name='fuse_{:d}'.format(i + 1))
                score = fused
            # 有点狠啊
            deconv_final = self.deconv2d(inputdata=score, out_channel=64, kernel_size=16,
                                         stride=8, use_bias=False, name='deconv_final')

            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')


            self._net_intermediate_results['binary_segment_logits'] = {
                'data': score_final,
                'shape': score_final.get_shape().as_list()
            }
            self._net_intermediate_results['instance_segment_logits'] = {
                'data': deconv_final,
                'shape': deconv_final.get_shape().as_list()
            }



    def build_model(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # mobilev2 fcn encode part
            self._mobilev2_fcn_encode(input_tensor=input_tensor, name='encode')
            # mobilev2 fcn decode part
            self._mobilev2_fcn_decode(name='decode')

        return self._net_intermediate_results


if __name__ == '__main__':
    """
    test code
    """
    test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    model = MOBILEV2FCN(phase='train')
    ret = model.build_model(test_in_tensor, name='mobilev2fcn')
    for layer_name, layer_info in ret.items():
        print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
