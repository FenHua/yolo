#!/usr/bin/python
#coding:utf-8
from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from bbox import bbox_iou

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) # numel()函数返回tensor中所有元素个数
def count_learnable_parameters(model):
    # 返回需要训练的参数数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def convert2cpu(matrix):
    # 根据GPU情况转换矩阵
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    # 在特征图上进行多尺度预测
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2) #步长
    grid_size = inp_dim // stride # cell大小
    bbox_attrs = 5 + num_classes # box属性数
    num_anchors = len(anchors) # 锚点数
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors] #将anchor大小归化到Cell中
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)# [batch_size,锚点数×(5+80),S*S] coco:80类
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs) 
    #Sigmoid函数处理中心点坐标和存在目标置信度将其归一化到[0,1]
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    #将当前cell坐标加到中心点
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset
    # 长宽对数空间转换
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    # 对类别进行sigmoid函数处理
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    prediction[:,:,:4] *= stride
    return prediction

def get_im_dim(im):
    # 得到图像的长宽
    im = cv2.imread(im)
    w, h = im.shape[1], im.shape[0]
    return w, h

def unique(tensor):
    tensor_np = tensor.cpu().numpy()    # 转换为numpy
    unique_np = np.unique(tensor_np)   # set操作
    unique_tensor = torch.from_numpy(unique_np) #转回
    tensor_result = tensor.new(unique_tensor.shape) # 构建[相同数据类型]的新Tensor
    tensor_result.copy_(unique_tensor) # 复制数据
    return tensor_result

def load_classes(namesfile):
    # 读取类名文件 coco：80 VOC：20
    # 返回: 元组,包括类名数组和总类的个数
    with open(namesfile, 'r') as f:
        names = f.read().split("\n")[:-1] # 最后一项是空行
        return len(names), names

def write_results(predictions, confidence, num_class, nms=True, nms_thresh=0.4):
    # 保留预测结果中置信度大于给定阈值的部分
    # confidence: shape=(1,10647, 85) 10647=3*(13*13+26*26+52*52)
    # mask: shape=(1,10647) => 增加一维度之后 (1, 10647, 1)
    mask = (predictions[:, :, 4] > confidence).float().unsqueeze(2) # 保留预测结果中置信度大于阈值的bbox
    predictions = predictions*mask # 小于置信度的条目值全为0, 剩下部分不变
    ind_nz = torch.nonzero(predictions[:, :, 4].squeeze()).squeeze() 
    if ind_nz.size(0) == 0:
        return 0 # 如果没有检测任何有效目标,返回值为0
    bbox = predictions.new(predictions.shape) # prediction的前五个数据分别表示 (Cx, Cy, w, h, score)
    # 将中心点和宽长度量方法改为左下右上坐标表示方法
    bbox[:, :, 0] = (predictions[:, :, 0] - predictions[:, :, 2]/2) # x1 = Cx - w/2
    bbox[:, :, 1] = (predictions[:, :, 1] - predictions[:, :, 3]/2) # y1 = Cy - h/2
    bbox[:, :, 2] = (predictions[:, :, 0] + predictions[:, :, 2]/2) # x2 = Cx + w/2
    bbox[:, :, 3] = (predictions[:, :, 1] + predictions[:, :, 3]/2) # y2 = Cy + h/2
    predictions[:, :, :4] = bbox[:, :, :4]  # 计算后的新坐标复制回去
    batch_size = predictions.size(0) # dim=0
    write = False # 拼接结果到output中最后返回
    for ind in range(batch_size):
        prediction = predictions[ind]  # 选择此batch中第ind个图像的预测结果
        ind_nz = torch.nonzero(prediction[:, 4].squeeze()).squeeze()
        if ind_nz.size(0) == 0:
            continue
        prediction = prediction[ind_nz, :] # shape=(10647->14, 85)
        max_score, max_score_ind = torch.max(prediction[:, 5:], 1) # 获取每一个候选框最可能的类型score和相应的index
        max_score = max_score.float().unsqueeze(1)  # 维度扩展，shape=(10647->14) => (10647->14,1)
        max_score_ind = max_score_ind.float().unsqueeze(1)
        seq = (prediction[:, :5], max_score, max_score_ind) # 获取有价值的信息，共6个
        prediction = torch.cat(seq, 1) 
        try:
            image_classes = unique(prediction[:, -1]) # 获取当前图像检测结果中出现的所有类别
        except:
            continue
        # 执行基于类别的NMS
        for cls in image_classes:
            class_mask = (prediction[:, -1] == cls)  # 分离检测结果中属于当前类的数据
            class_mask_ind = torch.nonzero(class_mask).squeeze() # shape=(n,1) => (n)
            prediction_class = prediction[class_mask_ind].view(-1, 7) # 从prediction中取出属于cls类别的所有结果，为下一步的nms的输入
            ''' 到此步 prediction_class 已经存在了我们需要进行非极大值抑制的数据 '''
            score_sort_ind = torch.sort(prediction_class[:, 4], descending=True)[1] # 返回排序索引
            prediction_class = prediction_class[score_sort_ind]
            cnt = prediction_class.size(0) # 个数
            '''开始执行 "非极大值抑制" 操作'''
            if nms:
                for i in range(cnt):
                    try:
                        ious = bbox_iou(prediction_class[i].unsqueeze(0), prediction_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:
                        break
                    iou_mask = (ious < nms_thresh).float().unsqueeze(1) # 计算出需要移除的item
                    prediction_class[i+1:] *= iou_mask # 保留i自身
                    non_zero_ind = torch.nonzero(prediction_class[:, 4].squeeze()) 
                    prediction_class = prediction_class[non_zero_ind].view(-1, 7) # 移除
            # 当前类的nms执行完之后，保存结果
            batch_ind = prediction_class.new(prediction_class.size(0), 1).fill_(ind)
            seq = batch_ind, prediction_class
            if  not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    return output
