# coding=utf-8
from __future__ import division, print_function
import tensorflow as tf
slim = tf.contrib.slim
from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer


def leaky_relu(inputs, alpha=0.1):
    # 泄露修正线性激活函数
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * inputs + f2 * tf.abs(inputs)

class yolov3(object):

    def __init__(self, class_num, anchors, batch_norm_decay=0.9):
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay

    def forward(self, inputs, is_training=False, reuse=False):
        self.img_size = tf.shape(inputs)[1:3]  # input大小:[1,416,416,3]
        # 设置 batch norm 参数
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm],reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: leaky_relu(x, 0.1)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list())
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list())
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.shape.as_list()[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # 根据feature map的大小resize锚点
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])
        # 根据最后一维分开feature map数据结果如下：
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)
        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)
        # 在feature map上得到绝对位置
        box_centers = box_centers + x_y_offset
        # 换算到原始图片上
        box_centers = box_centers * ratio[::-1]
        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
        # 相对原始图片大小
        box_sizes = box_sizes * ratio[::-1]
        # 拼接得到box信息
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])] #分层处理，不同大小的anchor在不同大小的feature map上滑动
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]
        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.shape.as_list()[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits
        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        # 从三个尺度上获取结果
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)
        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=2)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        '''
        grid_size = tf.shape(feature_map_i)[1:3]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)  # 下采样长宽尺度
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)  # N: batch_size
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)
        object_mask = y_true[..., 4:5]  # [N, 13, 13, 3, 1] 获取过滤器
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)  # 舍弃
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # 从feature map中得到xy坐标（0~1）
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset
        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))
        # box size punishment: 
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))
        # 损失函数部分
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N
        conf_pos_mask = object_mask  # shape: [N, 13, 13, 3, 1]
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N
        # 大小: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N
        return xy_loss, wh_loss, conf_loss, class_loss

    def compute_loss(self, y_pred, y_true):
        '''
        参数:
            y_pred: 返回feature_map表: [feature_map_1, feature_map_2, feature_map_3]
            y_true: 输入的原始label
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]
        # 计算三个层的loss
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        # 一种高效计算产生的box与正确box的IOU方法，这里这关注匹配面积
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]
        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)
        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)
        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
        return iou