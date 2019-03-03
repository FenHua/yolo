#!/usr/bin/python
#coding:utf-8
from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import preprocess_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools

def arg_parse():
    # 格式: 参数名, 目标参数(dest是字典的key),帮助信息,类型, 默认值
    parser = argparse.ArgumentParser(description="Yolo v3 检测模型") #创建一个解析器
    parser.add_argument("--images", dest="images", help="待检测图像目录", type=str, default="imgs")
    parser.add_argument("--dets", dest="dets", help="检测结果保存目录", type=str, default="det")
    parser.add_argument("--bs", dest="bs", help="Batch 大小", default=1)
    parser.add_argument("--confidence", dest="confidence", help="目标检测结果置信度阈值", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="非极大值抑制阈值", default=0.4)
    parser.add_argument("--cfg", dest="cfg", help="配置文件", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest="weights", help="检查点文件(权重文件)", default="yolov3.weights", type=str)
    parser.add_argument("--resize", dest="resize", help="网络输入分辨率. 分辨率越高,则准确率越高; 反之亦然.", \
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="缩放尺度", default="1,2,3", type=str)
    return parser.parse_args() # 返回转换好的结果，解析输入的命令行，参数默认是从sys.argv[1:]中获取，parse_args()返回一个命名空间

if __name__ == '__main__':
    args = arg_parse() 
    print(args)
    scales = args.scales
    images = args.images
    batch_size = args.bs
    confidence = args.confidence
    nms_thresh = args.nms_thresh
    CUDA = torch.cuda.is_available() # GPU环境是否可用
    num_class, classes = load_classes("data/coco.names") # COCO数据集
    model = Darknet(args.cfg) # 建立神经网络
    model.load_weights(args.weights)
    print("模型加载成功.")
    model.net_info["height"] = int(args.resize)  # 网络输入数据大小
    input_dim = model.net_info["height"]
    assert input_dim > 0 and input_dim % 32 == 0
    if CUDA:
        model.cuda() # 如果GPU可用, 模型切换到cuda中运行
    model.eval() # 评估模式
    try:
         # 加载待检测图像列表
        imlist = [os.path.join(images, img) for img in os.listdir(images) if os.path.splitext(img)[1] in [".png", ".jpg", ".jpeg"]]
    except NotADirectoryError:
        imlist = []
        imlist.append(os.path.join(images))
    except FileNotFoundError:
        print("%s 不是有效的目录或者文件" % (images,))
        exit()
    if not os.path.exists(args.dets):
        os.mkdir(args.dets)  # 存储结果目录
    load_batch = time.time()
    batches = list(map(preprocess_image, imlist, [input_dim for x in imlist]))  # 加载全部待检测图像
    ptt_images = [x[0] for x in batches] # ptt: pytorch_tensor
    ori_images = [x[1] for x in batches]
    ori_images_dim_list = [x[2] for x in batches]
    # repeat(*size), 沿着指定维度复制数据
    # 注: size维度必须和数据本身维度要一致
    ori_images_dim_list = torch.FloatTensor(ori_images_dim_list).repeat(1, 2) # (11,4) 原始图像尺寸
    if CUDA:
        ori_images_dim_list = ori_images_dim_list.cuda()
    # 所有检测结果
    # objs = []
    i = 0  # 第i个图像批次
    #  批处理 ...
    write = False
    if batch_size>1:
        # batch >1 支持实现
        num_batches = int(len(imlist)/batch_size + 0.5)
        ptt_images = [torch.cat(
                        (ptt_images[ i * batch_size: min((i + 1) * batch_size, len(ptt_images)) ]) )
                      for i in range(num_batches)]
    for batch in ptt_images:
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            predictions = model(Variable(batch), CUDA)
        predictions = write_results(predictions, confidence, num_class, nms=True, nms_thresh=nms_thresh) # 结果过滤
        if type(predictions) == int:
            i += 1
            continue
        end = time.time()
        print(end - start, predictions.shape) # 单位 秒
        predictions[:, 0] += i * batch_size # [0]表示图像索引
        if  not write:
            output = predictions
            write = True
        else:
            output = torch.cat((output, predictions))
        for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            im_ind =  i*batch_size + im_num # 图像编号
            objs = [ classes[int(x[-1])] for x in output if int(x[0]) == im_ind] # 输出第im_ind图像结果
            print("{0:20s} predicted in {1:6.3f} seconds".format(osp.split(image)[-1], (end-start)/batch_size))
            print("{0:20s} {1:s}".format("Objects detected", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1
    try:
        output  # 对所有的输入的检测结果
    except:
        print("没有检测到任何目标")
        exit()
    # 对output结果按照dim=0维度分组？
    ori_images_dim_list = torch.index_select(ori_images_dim_list, 0, output[:, 0].long()) # pytorch 切片torch.index_select(data, dim, indices)
    scaling_factor = torch.min(input_dim/ori_images_dim_list, 1)[0].view(-1, 1)
    # 坐标换算,将居中的位置坐标转换为以(0,0)为起点的坐标 x-x'soffset, y-y'soffset
    output[:, [1,3]] -= (input_dim - scaling_factor*ori_images_dim_list[:, 0].view(-1,1))/2 # x416 - (缩放后x<=416长度/2 )
    output[:, [2,4]] -= (input_dim - scaling_factor*ori_images_dim_list[:, 1].view(-1,1))/2
    output[:, 1:5] /= scaling_factor # 缩放至原图大小尺寸
    colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86)] #颜色
    def draw(x, batch, results):
        # 在图片上进行标注
        c1 = tuple(x[1:3].int())     # x1,y1
        c2 = tuple(x[3:5].int())     # x2,y2
        img = results[int(x[0])]    # 图像索引
        cls = int(x[-1])
        label = "%s" % classes[cls]  #类别信息
        color = random.choice(colors) # 随机选择颜色
        # 绘图
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img
    # 开始逐条绘制output中结果
    list(map(lambda x: draw(x, ptt_images, ori_images), output))
    # 保存文件路径
    det_names = ["{}/det_{}".format(args.dets, osp.split(x)[-1]) for x in imlist]
    list(map(cv2.imwrite, det_names, ori_images))
