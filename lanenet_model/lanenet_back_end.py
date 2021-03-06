#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_back_end.py
# @IDE: PyCharm
"""
LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
"""
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet_discriminative_loss
from semantic_segmentation_zoo import cnn_basenet

CFG = global_config.cfg


class LaneNetBackEnd(cnn_basenet.CNNBaseModel):
    """
    LaneNet backend branch which is mainly used for binary and instance segmentation loss calculation
    """

    def __init__(self, phase):
        """
        init lanenet backend
        :param phase: train or test
        """
        super(LaneNetBackEnd, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()

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

    @classmethod
    def _compute_class_weighted_cross_entropy_loss(cls, onehot_labels, logits, classes_weights):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :return:
        """
        loss_weights = tf.reduce_sum(tf.multiply(onehot_labels, classes_weights), axis=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels,
            logits=logits,
            weights=loss_weights
        )

        return loss

    def compute_loss(self, binary_seg_logits, binary_label,
                     instance_seg_logits, instance_label,
                     name, reuse, need_layer_norm=True):
        """
        compute lanenet loss
        :param binary_seg_logits: 256x512x2
        :param binary_label: 256x512x1
        :param instance_seg_logits: 256x512x64
        :param instance_label: # 256x512x1
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # calculate class weighted binary seg loss
            with tf.variable_scope(name_or_scope='binary_seg'):
                # binary_label_onehot = tf.one_hot(
                #     tf.reshape(
                #         tf.cast(binary_label, tf.int32),
                #         shape=[binary_label.get_shape().as_list()[0],
                #                binary_label.get_shape().as_list()[1],
                #                binary_label.get_shape().as_list()[2]]),
                #     depth=CFG.TRAIN.CLASSES_NUMS,
                #     axis=-1
                # ) # 256x512x1 -> 256x512x2(one-hot)

                binary_label_onehot = tf.one_hot(
                    tf.cast(binary_label, tf.int32)[:, :, :, 0],
                    depth=CFG.TRAIN.CLASSES_NUMS,
                    axis=-1
                )  # 256x512x1 -> 256x512x2(one-hot)

                # binary_label_plain = tf.reshape(
                #     binary_label,
                #     shape=[binary_label.get_shape().as_list()[0] *
                #            binary_label.get_shape().as_list()[1] *
                #            binary_label.get_shape().as_list()[2] *
                #            binary_label.get_shape().as_list()[3]])

                binary_label_plain = tf.reshape(binary_label, shape=[-1, ]) #
                unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
                counts = tf.cast(counts, tf.float32)  # 每个类别的像素数量
                inverse_weights = tf.divide(
                    1.0,
                    tf.log(tf.add(tf.divide(counts, tf.reduce_sum(counts)), tf.constant(1.02)))
                )  # 1/log(counts/all_counts + 1.02)

                binary_segmentation_loss = self._compute_class_weighted_cross_entropy_loss(
                    onehot_labels=binary_label_onehot,
                    logits=binary_seg_logits,
                    classes_weights=inverse_weights
                )

            # calculate class weighted instance seg loss
            with tf.variable_scope(name_or_scope='instance_seg'):
                if need_layer_norm:
                    instance_seg_logits = self.layerbn(
                        inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_bn = instance_seg_logits
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                pix_embedding = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )

                instance_segmentation_loss, l_var, l_dist, l_reg = \
                    lanenet_discriminative_loss.discriminative_loss(
                        pix_embedding, instance_label, CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                        delta_v=0.5, delta_d=3.0, param_var=1.0, param_dist=1.0, param_reg=0.001
                    )

            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'batchnorm' in vv.name or 'batch_norm' in vv.name \
                        or 'batch_normalization' in vv.name or 'gn' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = binary_segmentation_loss + instance_segmentation_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': binary_seg_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmentation_loss,
                'discriminative_loss': instance_segmentation_loss,
                'l2_reg_loss': l2_reg_loss
            }

        return ret

    def inference(self, binary_seg_logits, instance_seg_logits, name, reuse, need_layer_norm):
        """

        :param binary_seg_logits:
        :param instance_seg_logits:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            with tf.variable_scope(name_or_scope='binary_seg'):
                binary_seg_score = tf.nn.softmax(logits=binary_seg_logits)
                binary_seg_prediction = tf.argmax(binary_seg_score, axis=-1)

            with tf.variable_scope(name_or_scope='instance_seg'):
                if need_layer_norm:
                    instance_seg_logits = self.layerbn(
                        inputdata=instance_seg_logits, is_training=self._is_training, name='pix_bn')
                pix_bn = instance_seg_logits
                pix_relu = self.relu(inputdata=pix_bn, name='pix_relu')
                instance_seg_prediction = self.conv2d(
                    inputdata=pix_relu,
                    out_channel=CFG.TRAIN.EMBEDDING_FEATS_DIMS,
                    kernel_size=1,
                    use_bias=False,
                    name='pix_embedding_conv'
                )

        return binary_seg_prediction, instance_seg_prediction


if __name__ == '__main__':
    backend = LaneNetBackEnd(phase='train')
    binary_seg_logits = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 2], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_seg_logits = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 64], name='input')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    calculated_losses = backend.compute_loss(
        binary_seg_logits=binary_seg_logits,
        binary_label=binary_label,
        instance_seg_logits=instance_seg_logits,
        instance_label=instance_label,
        name='backend',
        reuse=tf.AUTO_REUSE
    )
