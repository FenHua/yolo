#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
import numpy as np
import yolo.config as cfg

class yolo_v2(object):
    def __init__(self, isTraining = True):
        self.classes = cfg.CLASSES # 类别名称
        self.num_class = len(self.classes) # 分类数
        self.box_per_cell = cfg.BOX_PRE_CELL # 每个单元格预测的box数
        self.cell_size = cfg.CELL_SIZE # 图像划分大小
        self.batch_size = cfg.BATCH_SIZE # batch大小
        self.image_size = cfg.IMAGE_SIZE # 图片大小
        self.anchor = cfg.ANCHOR # anchor 数组
        self.alpha = cfg.ALPHA # 泄露线性函数系数 默认0.1
        self.class_scale = 1.0 #loss函数中类别损失权重
        self.object_scale = 5.0 # loss函数中有对象时的预测损失权重
        self.noobject_scale = 1.0 #loss函数中没有对象时预测损失权重
        self.coordinate_scale = 1.0 # loss函数中坐标损失权重
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        self.offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32), [1, self.cell_size, self.cell_size, self.box_per_cell])
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1)) # 第一cell的位置坐标，偏置
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images') #输入图片
        self.logits = self.build_networks(self.images) #网络输出
        if isTraining:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5], name = 'labels') #S*S**B*(5+C)
            self.total_loss = self.loss_layer(self.logits, self.labels) # 带有各部分权重的损失函数
            tf.summary.scalar('total_loss', self.total_loss) # 将总的损失写入日志
            
    def build_networks(self, inputs):
        # 假设输入图片大小为416*416
        net = self.conv_layer(inputs, [3, 3, 3, 32], name = '0_conv')
        net = self.pooling_layer(net, name = '1_pool') # 208×208*32
        net = self.conv_layer(net, [3, 3, 32, 64], name = '2_conv')
        net = self.pooling_layer(net, name = '3_pool') # 104*104*64
        net = self.conv_layer(net, [3, 3, 64, 128], name = '4_conv')
        net = self.conv_layer(net, [1, 1, 128, 64], name = '5_conv')
        net = self.conv_layer(net, [3, 3, 64, 128], name = '6_conv')
        net = self.pooling_layer(net, name = '7_pool') #52*52*128
        net = self.conv_layer(net, [3, 3, 128, 256], name = '8_conv')
        net = self.conv_layer(net, [1, 1, 256, 128], name = '9_conv')
        net = self.conv_layer(net, [3, 3, 128, 256], name = '10_conv')
        net = self.pooling_layer(net, name = '11_pool') #26*26*256
        net = self.conv_layer(net, [3, 3, 256, 512], name = '12_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], name = '13_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], name = '14_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], name = '15_conv')

        net16 = self.conv_layer(net, [3, 3, 256, 512], name = '16_conv')

        net = self.pooling_layer(net16, name = '17_pool') #13*13*512
        net = self.conv_layer(net, [3, 3, 512, 1024], name = '18_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512], name = '19_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], name = '20_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512], name = '21_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], name = '22_conv')
        net = self.conv_layer(net, [3, 3, 1024, 1024], name = '23_conv')
        
        net24 = self.conv_layer(net, [3, 3, 1024, 1024], name = '24_conv') #13*13*1024
        #passthrough处理
        net = self.conv_layer(net16, [1, 1, 512, 64], name = '26_conv') # 26*26*64
        net = self.reorg(net) # passthrough处理，进行Fine-Grained Feature 得到13*13*256
        net = tf.concat([net, net24], 3) # 连接后得到13*13×（1024+256）
        net = self.conv_layer(net, [3, 3, int(net.get_shape()[3]), 1024], name = '29_conv') # 13*13*1024
        # 去掉batch_normalation 最终得到 13×13×125
        net = self.conv_layer(net, [1, 1, 1024, self.box_per_cell * (self.num_class + 5)], batch_norm=False, name = '30_conv') 

        return net


    def conv_layer(self, inputs, shape, batch_norm = True, name = '0_conv'):
        # 在每个卷积层后面添加一个batch normalizition
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight') #卷积核权重
        biases = tf.Variable(tf.constant(0.1, shape=[shape[3]]), name='biases') # 偏置项，后面激活函数使用
        conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name) #卷积时进行pad填充
        if batch_norm:
            depth = shape[3] # 核的数量大小(深度)
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale') # NB算法中的belta
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift') # NB算法中的gamma
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean') #均值
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance') #方差
            conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05) # 固定在一定区间，且具备足够的非线性表达
            conv = tf.add(conv_bn, biases)
            conv = tf.maximum(self.alpha * conv, conv) # 泄露线性修正函数
        else:
            conv = tf.add(conv, biases)
        return conv

    def pooling_layer(self, inputs, name = '1_pool'):
        # 最大池化层 大小2*2 步长2
        pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
        return pool


    def reorg(self, inputs):
        '''
        passthrough中的重组层
        主要利用Fine-Grained Feature，细颗粒特征用于检测微小物体
        example:
        inputs[：，：：2，：：2，：]
        意思是第一维度，从开始以步长为1到结束
        第二维度，从开始以步长为2到结束输出
        第三维度，从开始以步长为2到结束输出
        第四维度，从开始以步长为1到结束
        '''
        outputs_1 = inputs[:, ::2, ::2, :]
        outputs_2 = inputs[:, ::2, 1::2, :] # 1表示从下坐标1开始
        outputs_3 = inputs[:, 1::2, ::2, :]
        outputs_4 = inputs[:, 1::2, 1::2, :]
        output = tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis = 3) #返回一个与input具有相同的类型的tensor
        return output


    def loss_layer(self, predict, label):
        # 预测输出形式[batch_size,S,S,B,(5+C)] 
        predict = tf.reshape(predict, [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5])
        # 预测框坐标 
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        # 每个预测box的置信度，表示有无目标
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 1])
        # 每个box所属类别信息
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])
        # box实际坐标信息的转换计算计算，中心点坐标比例加偏置，长宽比例变换
        boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + self.offset) / self.cell_size,
                           (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(self.offset, (0, 2, 1, 3))) / self.cell_size,
                           tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 1, 5]) / self.cell_size),
                           tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 1, 5]) / self.cell_size)])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))
        # sigmoid函数将置信度固定在0~1之间
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        #网络最终输出的是score，通过softmax转换为概率
        box_classes = tf.nn.softmax(box_classes)
        #每个box是否含有目标
        response = tf.reshape(label[:, :, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell])
        #目标坐标信息
        boxes = tf.reshape(label[:, :, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        #目标类别信息
        classes = tf.reshape(label[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])
        iou = self.calc_iou(box_coor_trans, boxes) # 计算IOU
        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))#返回每个cell的最佳box
        confs = tf.expand_dims(best_box * response, axis = 4) # 分对
        '''有无目标，坐标，类别预测正确与否的index'''
        conid = self.noobject_scale * (1.0 - confs) + self.object_scale * confs
        cooid = self.coordinate_scale * confs
        proid = self.class_scale * confs
        coo_loss = cooid * tf.square(box_coor_trans - boxes) # 坐标loss和
        con_loss = conid * tf.square(box_confidence - confs) # 有无目标loss和
        pro_loss = proid * tf.square(box_classes - classes)# 类别loss
        loss = tf.concat([coo_loss, con_loss, pro_loss], axis = 4) #loss加一起
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis = [1, 2, 3, 4]), name = 'loss')
        return loss
    
    def calc_iou(self, boxes1, boxes2):
        # 计算IOU
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]# box1的面积
        box = tf.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])#左上角与右下角坐标
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]# box2面积
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))#左上角与右下角坐标
        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2]) #最大左上点
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:]) #最小右下角
        intersection = tf.maximum(right_down - left_up, 0.0) #交
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        union_square = boxes1_square + boxes2_square - inter_square #交面
        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)
