# coding: utf-8
from __future__ import division, print_function
import cv2
import random

def get_color_table(class_num, seed=2):
    # 字典类型，一种类别对应一种颜色
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table

def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    # 画框并标记，参数line_thickness: 线条的粗细 int类型
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # 线条粗细
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])) #坐标
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # 画框
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

