#!/usr/bin/python
#coding:utf-8
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from matplotlib import pyplot as plt

class Detector(object):
    # 目标检测
    def __init__(self, net, weight_file):
        self.net = net  # yolo网络
        self.weights_file = weight_file  # 检查点文件路径
        self.output_dir = os.path.dirname(self.weights_file) #输出文件夹路径
        self.classes = cfg.CLASSES  # 数据集类别名
        self.num_class = len(self.classes) # 类别数
        self.image_size = cfg.IMAGE_SIZE # 图像大小
        self.cell_size = cfg.CELL_SIZE # 单元格大小
        self.boxes_per_cell = cfg.BOXES_PER_CELL #  每个网络边界框的个数B=2
        self.threshold = cfg.THRESHOLD # 目标存在阈值
        self.iou_threshold = cfg.IOU_THRESHOLD # NMS阈值
        self.boundary1 = self.cell_size * self.cell_size * self.num_class #所有cell所对应的类别预测
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell
        # 所有类别之后每个cell所对应的bounding boxes的数量总和
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) # 运行图之前初始化变量
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(self.output_dir)  #直接载入最近保存的检查点文件
        print("ckpt:",ckpt)         
        if ckpt!=None:
            self.saver.restore(self.sess, ckpt)  #如果存在检查点文件 则恢复模型
        else:
            self.saver.restore(self.sess, self.weights_file)  #从指定检查点文件恢复

    def draw_result(self, img, result):
        # 在原图上绘制边界框
        for i in range(len(result)):
            x = int(result[i][1]) # x_center
            y = int(result[i][2]) # y_center
            w = int(result[i][3] / 2) #w/2
            h = int(result[i][4] / 2)  #h/2
            # 绘制目标边界框，矩形左上右下角
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1) # 绘制矩形框存放类别名
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA #线性
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType) # 绘制文本信息(类别名和置信度)

    def detect(self, img):
        '''
        图片目标检测，返回检测到的边界框
        list类型，每个元素对应一个目标框[类别名，x_center,y_center,w,h,置信度]
        '''
        img_h, img_w, _ = img.shape # 图片高和宽
        inputs = cv2.resize(img, (self.image_size, self.image_size)) # 图片缩放
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32) # RGB->RGB
        inputs = (inputs / 255.0) * 2.0 - 1.0 # 归一化处理
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3)) #reshape [1,448,448,3]
        result = self.detect_from_cvmat(inputs)[0] # 获取网络输出第一项[1,1470]
        for i in range(len(result)):
            # x_center,y_center,w,h都是真实值，分别表示预测边界框的中心坐标，宽，高
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)
        return result

    def detect_from_cvmat(self, inputs):
        # 检测
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i])) # 获取有目标的边界框
        return results

    def interpret_output(self, output):
        '''
        对网络输出结果进行处理，并记录有目标的边界框
        output:
        yolo网络输出的每一行数据大小[1470,]
        0:7*7*20 预测类别
        7*7*20:7*7*20+7*7*2 表示预测置信度，即预测的边界框与实际边界框的IOU
        7*7*20+7*7*2:1470  预测边界框，中心点相对于当前格子的，宽度和高度的开根号
        相对于当前整张图像的(归一化)
        函数返回目标检测的边界框，其中置信度是网络输出置信度与类别概率的乘积
        '''
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class)) #概率[7,7,2,20]
        class_probs = np.reshape(output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class)) # 类别概率[7,7,20]
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell)) # box是否含目标[7,7,2]
        boxes = np.reshape(output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4)) # 边界框[7,7,2,4]
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell) #[14,7]
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0)) #[7,7,2] 每一行都是[[0,0],[1,1],[2,2]....[6,6]]
        # 目标中心相对于cell而言
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:]) # 宽度和高度相对于整个图片
        boxes *= self.image_size #转换为实际的编辑框
        for i in range(self.boxes_per_cell):
            # 遍历每一个边界框的置信度
            for j in range(self.num_class):
                # 遍历每一个类别
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i]) # 在测试时，条件类概率和单个
        # 盒子的置信度相乘来判断j类出现在框i中的概率以及预测框拟合目标的程度
        
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool') # 如果存在目标，则标识1，否则0
        filter_mat_boxes = np.nonzero(filter_mat_probs) # 返回非0值的索引
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]] #获取有目标的盒子
        probs_filtered = probs[filter_mat_probs] #目标盒子的置信度
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]] #获取检测到目标的边界框的置信度[n,]
        argsort = np.array(np.argsort(probs_filtered))[::-1] # 按照置信度倒序排序，返回对应的索引
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    # 非极大值抑制
                    probs_filtered[j] = 0.0
        # 非极大值抑制
        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
        #整理结果
        result = []
        for i in range(len(boxes_filtered)):
            #记录每一个边界框的信息[类别名，x中心，y中心，宽度，高度，置信度]
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])
        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, cap, wait=10):
        # 调用摄像头，实时检测
        detect_timer = Timer()
        ret, _ = cap.read() #读取一帧
        while ret:
            ret, frame = cap.read() # 读取一帧
            detect_timer.tic() 
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))
            self.draw_result(frame, result) # 绘制边界框 添加附加信息
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)
            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        # 对图片目标进行检测
        detect_timer = Timer()
        image = cv2.imread(imname)
        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))
        self.draw_result(image, result)
        image=image[:,:,::-1]
        plt.imshow(image)
        plt.show()
        cv2.waitKey(wait)