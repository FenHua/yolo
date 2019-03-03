# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def conv2d(inputs, filters, kernel_size, strides=1):

    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],[pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    if strides > 1: 
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)
        net = net + shortcut
        return net
    # 开始两层卷积
    net = conv2d(inputs, 32,  3, strides=1)
    net = conv2d(net, 64,  3, strides=2)
    # res_block *1
    net = res_block(net, 32)
    net = conv2d(net, 128, 3, strides=2)
    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)
    net = conv2d(net, 256, 3, strides=2)
    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)
    route_1 = net # 特征层1
    net = conv2d(net, 512, 3, strides=2)
    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)
    route_2 = net #特征层2
    net = conv2d(net, 1024, 3, strides=2)
    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net #特征层3
    return route_1, route_2, route_3

def yolo_block(inputs, filters):
    # yolo 网络模型块
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net

def upsample_layer(inputs, out_shape):
    # 上采样层
    new_height, new_width = out_shape[1], out_shape[2]
    # 注意，长优先
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    return inputs
