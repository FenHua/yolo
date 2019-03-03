# coding: utf-8
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

input_image = "/home/yhq/Desktop/yolo_v3/data/demo_data/kite.jpg"  # 输入测试图片
anchor_path = "/home/yhq/Desktop/yolo_v3/data/yolo_anchors.txt"  # anchor文件路径
new_size = [416, 416]  # 模型输入图片大小
class_name_path = "/home/yhq/Desktop/yolo_v3/data/coco.names"  # 类别名称文件路径
restore_path = "/home/yhq/Desktop/yolo_v3/data/darknet_weights/yolov3.ckpt"  # 检查点文件
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)
color_table = get_color_table(num_class)  # 每个类别对应一个color
# 读图片并进行一些简单处理
img_ori = cv2.imread(input_image)
height_ori, width_ori = img_ori.shape[:2]
img = cv2.resize(img_ori, tuple(new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.
# 识别工作
with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
    yolo_model = yolov3(num_class, anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    saver = tf.train.Saver()  # 变量初始化后建立
    saver.restore(sess, restore_path)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.4, iou_thresh=0.5)
    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
    # 将坐标信息转换到原始图片上
    boxes_[:, 0] *= (width_ori/float(new_size[0]))
    boxes_[:, 2] *= (width_ori/float(new_size[0]))
    boxes_[:, 1] *= (height_ori/float(new_size[1]))
    boxes_[:, 3] *= (height_ori/float(new_size[1]))
    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey()