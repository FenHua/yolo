#!/usr/bin/python
#coding:utf-8
from __future__ import division
import torch 
import random
import numpy as np
import cv2

def confidence_filter(result, confidence):
    # 返回score值大于阈值的结果
    conf_mask = (result[:,:,4] > confidence).float().unsqueeze(2)
    result = result*conf_mask    
    return result

def confidence_filter_cls(result, confidence):
    max_scores = torch.max(result[:,:,5:25], 2)[0] #获取最大score的类
    res = torch.cat((result, max_scores),2) #拼接
    print(res.shape)
    cond_1 = (res[:,:,4] > confidence).float()
    cond_2 = (res[:,:,25] > 0.995).float()
    conf = cond_1 + cond_2
    conf = torch.clamp(conf, 0.0, 1.0) #clamp  锁住 堆
    conf = conf.unsqueeze(2)
    result = result*conf   #返回IOU大于阈值或者类别score大于0.995的结果
    return result

def get_abs_coord(box):
    # 返回候选框绝对坐标系下的坐标
    box[2], box[3] = abs(box[2]), abs(box[3])
    x1 = (box[0] - box[2]/2) - 1 
    y1 = (box[1] - box[3]/2) - 1 
    x2 = (box[0] + box[2]/2) - 1 
    y2 = (box[1] + box[3]/2) - 1
    return x1, y1, x2, y2
    
def sanity_fix(box):
    # 判断 修正，保证左下小于右上
    if (box[0] > box[2]):
        box[0], box[2] = box[2], box[0]
    if (box[1] >  box[3]):
        box[1], box[3] = box[3], box[1]
    return box

def bbox_iou(box1, box2):
    # 返回俩候选框的IOU
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    # 获取两个候选框的交坐标
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # 交的面积
    if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))
    # 两个候选框的面积
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area) #IOU
    return iou

def pred_corner_coord(prediction):
    ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous() # 获取非0置信度的候选框
    box = prediction[ind_nz[0], ind_nz[1]]
    box_a = box.new(box.shape)
    box_a[:,0] = (box[:,0] - box[:,2]/2)
    box_a[:,1] = (box[:,1] - box[:,3]/2)
    box_a[:,2] = (box[:,0] + box[:,2]/2) 
    box_a[:,3] = (box[:,1] + box[:,3]/2)
    box[:,:4] = box_a[:,:4]
    prediction[ind_nz[0], ind_nz[1]] = box #重写
    return prediction

def write(x, batches, results, colors, classes):
    # 在原图上进行标注操作
    # 坐标大小
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1]) # 类别信息
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1) #标注类别信息
    return img
