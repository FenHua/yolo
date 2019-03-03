# coding: utf-8

from __future__ import division, print_function
import numpy as np
import tensorflow as tf

def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    """
    在GPU上进行NMS
    参数：
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: 总的类别数
        max_boxes: 整形，希望预测的最大box数量
        score_thresh: 置信度阈值
        iou_thresh: IOU阈值
    """
    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')
    boxes = tf.reshape(boxes, [-1, 4])  # ‘-1’表示不知道boxes的个数，程序自动计算
    score = tf.reshape(scores, [-1, num_classes])
    mask = tf.greater_equal(score, tf.constant(score_thresh))# Step 1: 根据置信度获得一个过滤网
    for i in range(num_classes):
        # Step 2: 基于类别的NMS
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])  # 过滤后的box
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=iou_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))
    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)
    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    # 基本的NMS，其中boxes: shape of [-1, 4]，scores: shape of [-1,]，max_boxes: 表示通过NMS后选择的最大盒子数量
    assert boxes.shape[1] == 4 and len(scores.shape) == 1
    # box坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1] # 对分数进行下降排序
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h #俩box相交的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # iou
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, iou_thresh=0.5):
    # 在CPU下执行NMS，其中boxes: shape [1, 10647, 4]，scores: shape [1, 10647, num_classes]
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # 提取信息
    picked_boxes, picked_score, picked_label = [], [], []
    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)  # 类型i下的所有目标
        filter_boxes = boxes[indices]  # 类型i下的box
        filter_scores = scores[:,i][indices]  # 只考虑当前类别下的score
        if len(filter_boxes) == 0: 
            continue
        # NMS
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: 
        return None, None, None
    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)
    return boxes, score, label