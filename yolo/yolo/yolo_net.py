#!/usr/bin/python
#coding:utf-8
import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim
class YOLONet(object):
    
    def __init__(self, is_training=True):
        # 构造函数，利用config中的信息
        self.classes = cfg.CLASSES  # 类别名称
        self.num_class = len(self.classes)     # 类别数
        self.image_size = cfg.IMAGE_SIZE  # 输入图片大小，448*448
        self.cell_size = cfg.CELL_SIZE          # 单元格的大小，7*7
        self.boxes_per_cell = cfg.BOXES_PER_CELL # 每个单元格预测框的个数
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)  # 网络输出的大小：S×S×（5×B+C）=1470
        self.scale = 1.0 * self.image_size / self.cell_size # 特征到原图放缩比例
        self.boundary1 = self.cell_size * self.cell_size * self.num_class  # 输出结果类别信息7*7*20
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell # 输出结果类别信息和单元格每个盒子置信度
        self.object_scale = cfg.OBJECT_SCALE            # loss函数有目标权重
        self.noobject_scale = cfg.NOOBJECT_SCALE   # loss函数没有目标权重
        self.class_scale = cfg.CLASS_SCALE                 # loss函数类别信息权重
        self.coord_scale = cfg.COORD_SCALE             # loss函数坐标框权重
        self.learning_rate = cfg.LEARNING_RATE       # 学习率0.0001
        self.batch_size = cfg.BATCH_SIZE                   # batch大小 
        self.alpha = cfg.ALPHA                                     # 泄露修正线性激活函数系数0.1
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))   # 偏置
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images') # 输入图片
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training) # 构建网络输出参数[None,1470]
        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class]) # 标签占位符[None,S,S,5+C]
            self.loss_layer(self.logits, self.labels)        # 设置损失函数
            self.total_loss = tf.losses.get_total_loss() #  加入权重的损失函数
            tf.summary.scalar('total_loss', self.total_loss) # 将损失以标量形式显示，该变量命名为total_loss
            
    def build_network(self, images, num_outputs, alpha, keep_prob=0.5, is_training=True, scope='yolo'):
        # keep_prob 保留率  scope 命名空间名
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1') # pad_1填充454*454*3
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2') # 卷积层 s=2 224*224*64
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3') #池化层 112*112*64
                net = slim.conv2d(net, 192, 3, scope='conv_4') #卷积层 112*112*192
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5') #池化层 56*56*192
                net = slim.conv2d(net, 128, 1, scope='conv_6') #卷积层 56*56*128
                net = slim.conv2d(net, 256, 3, scope='conv_7') #卷积层 56*56*256
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9') #卷积层 56*56*512
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10') #池化层 28*28*512
                net = slim.conv2d(net, 256, 1, scope='conv_11') #卷积层 28*28*256
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20') #卷积层 28*28*1024
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21') #14*14*1024
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26') #14*14*1024 
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),name='pad_27') # 填充 16*16*1024
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28') # 7×7*1024 
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30') # 7*7*1024
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31') # 转置[None,1024,7,7]
                net = slim.flatten(net, scope='flat_32') #展开 50176
                net = slim.fully_connected(net, 512, scope='fc_33') # 全连接fc_33 512 
                net = slim.fully_connected(net, 4096, scope='fc_34') #全连接 fc_34 4096
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36') #全连接层 fc_36 1470
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
       # 计算IOU
        with tf.variable_scope(scope):
            # 将坐标从 (x_center, y_center, w, h) 转到 (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)
            #[左上角x，左上角y，右下角x，右下角y]
            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)
            # 计算左上角和右下角的坐标
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])
            # 交
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]
            # 并
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]
            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)
        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        '''
        损失函数
        predicts:网络输出为1470维，其中0:7×7*20：表示预测类别
        7*7*20:7*7*20+7*7*2：表示预测置信度，即预测的边界框与实际框之间的IOU
        7*7*20+7*7*2:1470：预测边界框 目标中心是相对于当前的格子，宽度和高度
        的开根号是相对于整副图像的
        labels：标签值，形状[None 7,7,25],0:1置信度，表示有没有目标；1:5目标边界框
        5:25 目标类别
        '''
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class]) # 预测每个格子目标的类别
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell]) # 预测每个格子两个边界框的置信度
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4]) # 预测边界框
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1]) #每个单元格是否含目标
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4]) #每个格子中目标的坐标
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]
            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            # 假设当前格子为(3,3)，当前格子的预测边界框为(x0,y0)，则计算坐标(x,y) = ((x0,y0)+(3,3))/7
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes) #计算IOU
            # object_mask就表示每个格子中的哪个边界框负责该格子中目标预测？
            #哪个边界框取值为1，哪个边界框就负责目标预测
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response
            # noobject_mask就表示每个边界框不负责该目标的置信度
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask
            # 将之前的坐标换回来(相对整个图像->相对当前的格子)，长和宽开方
            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)
            # 分类损失
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale
            # 有目标物体存在的置信度预测损失
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale
            # 没有目标物体存在的置信度的损失
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale
            # 边界框坐标损失
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale
            # 将所有的loss加在一起
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)
            # 将所有训练信息写入日志
            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)
            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)

def leaky_relu(alpha):
    # 泄露修正线性激活函数
    def op(inputs):
        with tf.variable_scope('leaky_relu'):
         f1 = 0.5 * (1 + alpha)
         f2 = 0.5 * (1 - alpha)
         return f1 * inputs + f2 * tf.abs(inputs)  
    return op
