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
from util import count_parameters as count
from util import convert2cpu as cpu
from PIL import Image, ImageDraw

def letterbox_image_(img, input_dim):
    # 双线性插值缩放
    return cv2.resize(src=img, dsize=(input_dim[1], input_dim[0])) # 默认的双线性插值

def letterbox_image(img, input_dim):
    # 调整图像尺寸
    ori_w, ori_h = img.shape[1], img.shape[0]
    dst_w, dst_h = input_dim
    new_w = int(ori_w * min(dst_w/ori_w, dst_h/ori_h)) # 保证长宽至少一个正好等于目标长度
    new_h = int(ori_h * min(dst_w/ori_w, dst_h/ori_h))
    resized_image = cv2.resize(src=img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC) # 双三次插值
    # 创建一个画布, 将resized_image数据 拷贝到画布中心
    canvas = np.full(shape=(dst_w, dst_h, 3), fill_value=128)
    canvas[(dst_h-new_h)//2: (dst_h-new_h)//2 + new_h, (dst_w-new_w)//2: (dst_w-new_w)//2 + new_w, :] = resized_image
    return canvas

def preprocess_image(img, input_dim):
    # 处理输入图像数据，返回值: 处理后图像, 原图, 原图尺寸
    ori_img = cv2.imread(img)
    dim = ori_img.shape[1], ori_img.shape[0] # 长宽
    img = letterbox_image(ori_img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, ori_img, dim

def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(3,*network_dim).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3,*network_dim)
    img = img.float().div(255.0)
    return (img, orig_im, dim)

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)
    inp = inp[:,:,::-1] #转换，便于显示
    return inp
