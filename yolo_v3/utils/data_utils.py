# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2

def parse_line(line):
    # 从训练或者测试txt文件中获取图片路径，box信息，类别信息
    s = line.strip().split(' ')
    pic_path = s[0]
    s = s[1:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return pic_path, boxes, labels

def resize_image_and_correct_boxes(img, boxes, img_size):
    # 将图片从灰度图转为3通道图片
    if len(img) == 2:
        img = np.expand_dims(img, -1)
    ori_height, ori_width = img.shape[:2]
    new_width, new_height = img_size
    img = cv2.resize(img, (new_width, new_height))
    img = np.asarray(img, np.float32)  # 转为float
    boxes[:, 0] = boxes[:, 0] / ori_width * new_width
    boxes[:, 2] = boxes[:, 2] / ori_width * new_width
    boxes[:, 1] = boxes[:, 1] / ori_height * new_height
    boxes[:, 3] = boxes[:, 3] / ori_height * new_height
    return img, boxes

def data_augmentation(img, boxes, label):
    '''
    数据增强部分
        img: [H, W, 3] RGB形式
        boxes: [N, 4]  N 真实的box数量,box存储形式[x_min, y_min, x_max, y_max]
        label: [N]
    '''
    return img, boxes, label


def process_box(boxes, labels, img_size, class_num, anchors):
    # 产生真实类别标签对应于三个不同大小的feature map
    anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2  # 获取box的中心点
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]  # 获取box的长宽
    y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 5 + class_num), np.float32)
    y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 5 + class_num), np.float32)
    y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 5 + class_num), np.float32)
    y_true = [y_true_13, y_true_26, y_true_52]
    box_sizes = np.expand_dims(box_sizes, 1)  # [N, 1, 2]
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    mins = np.maximum(- box_sizes / 2, - anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    whs = maxs - mins # [N, 9, 2]
    # [N, 9]
    iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1] + 1e-10)
    best_match_idx = np.argmax(iou, axis=1)  # [N]
    ratio_dict = {1.: 8., 2.: 16., 3.: 32.}
    for i, idx in enumerate(best_match_idx):
        feature_map_group = 2 - idx // 3  # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]  # 尺度比率: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]
        # print feature_map_group, '|', y,x,k,c
        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5+c] = 1.
    return y_true_13, y_true_26, y_true_52

def parse_data(line, class_num, img_size, anchors, mode):
    '''
    param:
        line: 训练或者测试文件路经
        args: args returned from the main program
        mode: 'train' 或者 'val'. 当设置为 'train', 自动进行数据增强
    '''
    pic_path, boxes, labels = parse_line(line)  # 从训练或者测试txt文件中获取图片路径，box信息，类别信息
    img = cv2.imread(pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)
    if mode == 'train':
        img, boxes, labels = data_augmentation(img, boxes, labels)  # 数据增强
    img = img / 255.  # 输入模型的数据应该在0~1
    y_true_13, y_true_26, y_true_52 = process_box(boxes, labels, img_size, class_num, anchors)
    return img, y_true_13, y_true_26, y_true_52
