#!/usr/bin/python
#coding:utf-8
import torch
import numpy as np
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    # 在特整图上进行多尺度预测, 在每个Cell都有三个不同尺度的锚点
    batch_size = prediction.size(0) #batch_size
    stride = inp_dim // prediction.size(2) # 步长
    grid_size = inp_dim // stride #cell大小
    bbox_attrs = 5 + num_classes # box包含[5+类别数]
    num_anchors = len(anchors) # 锚点数
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors] #将anchor大小归化到Cell中
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    # Sigmoid函数处理中心点和存在目标置信度将其归一化到[0,1]
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    # 将当前cell左上角坐标与当前的中心点坐标相加
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    anchors = torch.FloatTensor(anchors) # 长宽对数空间转换
    if CUDA:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes])) # 对类别进行sigmoid函数处理
    prediction[:, :, :4] *= stride
    return prediction
