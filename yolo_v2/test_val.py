#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
import numpy as np
import argparse
import colorsys
import cv2
import os

import yolo.config as cfg
from yolo.yolo_v2 import yolo_v2
# from yolo.darknet19 import Darknet19

class Detector(object):
    def __init__(self, yolo, weights_file):
        self.yolo = yolo #yolo模型
        self.classes = cfg.CLASSES #拟识别类名称
        self.num_classes = len(self.classes) #类别数
        self.image_size = cfg.IMAGE_SIZE # 图像大小
        self.cell_size = cfg.CELL_SIZE # 单元格大小
        self.batch_size = cfg.BATCH_SIZE #batch大小
        self.box_per_cell = cfg.BOX_PRE_CELL #每个单元格预测box的数量
        self.threshold = cfg.THRESHOLD # 判断目标阈值
        self.anchor = cfg.ANCHOR # anchor比例数组
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer()) #tensorflow变量初始化
        print('Restore weights from: ' + weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weights_file) # 从指定检查点文件恢复模型

    def detect(self, image):
        '''
        图片目标检测，返回检测到的边界框
        list类型，每个元素对应一个目标框[类别名，x_center,y_center,w,h,置信度]
        '''
        image_h, image_w, _ = image.shape # 图片高和宽
        image = cv2.resize(image, (self.image_size, self.image_size)) #缩放，便于网络使用
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) #颜色空间转换
        image = image / 255.0 * 2.0 - 1.0 #归一化
        image = np.reshape(image, [1, self.image_size, self.image_size, 3]) #[1,416,416,3]
        output = self.sess.run(self.yolo.logits, feed_dict = {self.yolo.images: image}) #获取网络输出
        results = self.calc_output(output) #获取检测目标信息
        for i in range(len(results)):
            # 位置信息在原始图片上的映射
            results[i][1] *= (1.0 * image_w / self.image_size)
            results[i][2] *= (1.0 * image_h / self.image_size)
            results[i][3] *= (1.0 * image_w / self.image_size)
            results[i][4] *= (1.0 * image_h / self.image_size)
        return results

    def calc_output(self, output):
        # 从网络输出中整理出目标
        output = np.reshape(output, [self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        boxes = np.reshape(output[:, :, :, :4], [self.cell_size, self.cell_size, self.box_per_cell, 4])    #boxes 坐标
        boxes = self.get_boxes(boxes) * self.image_size # boxes 在网络输入图像上的坐标
        confidence = np.reshape(output[:, :, :, 4], [self.cell_size, self.cell_size, self.box_per_cell])    #每个box的置信度(区别与V1)
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence)) #  置信度0~1
        # 类别概率的计算 当前回归框内类别情况
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, self.num_classes)) 
        classes = np.reshape(output[:, :, :, 5:], [self.cell_size, self.cell_size, self.box_per_cell, self.num_classes])    #类别
        classes = np.exp(classes) / np.tile(np.expand_dims(np.sum(np.exp(classes), axis=3), axis=3), (1, 1, 1, self.num_classes))
        probs = classes * confidence #类条件概率
        filter_probs = np.array(probs >= self.threshold, dtype = 'bool') #存在目标
        filter_index = np.nonzero(filter_probs) #分别统计三个维度中非零坐标
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]] #得到有目标的box
        probs_filter = probs[filter_probs] # 含有目标的概率
        classes_num = np.argmax(filter_probs, axis = 3)[filter_index[0], filter_index[1], filter_index[2]]
        sort_num = np.array(np.argsort(probs_filter))[::-1] #根据置信度倒排
        box_filter = box_filter[sort_num] # 过滤后含目标的box
        probs_filter = probs_filter[sort_num] # 过滤后含目标box的置信度
        classes_num = classes_num[sort_num] # 含目标box的类别
        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if self.calc_iou(box_filter[i], box_filter[j]) > 0.5:
                    # NMS
                    probs_filter[j] = 0.0
        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        # 最终具有目标的box信息
        probs_filter = probs_filter[filter_probs] 
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]
        # 记录检测结果[类别，回归框坐标信息，置信度]
        results = []
        for i in range(len(probs_filter)):
            results.append([self.classes[classes_num[i]], box_filter[i][0], box_filter[i][1],
                            box_filter[i][2], box_filter[i][3], probs_filter[i]])
        return results

    def get_boxes(self, boxes):
        '''
        获取回归框像素级别的表示，每个格栅预测5的box，对应5个不同的anchor
        为了将bounding box的中心点约束在当前cell中，使用sigmoid函数将tx、ty归一化处理
        将值约束在0~1，这使得模型训练更稳定。
        通过预测的四个值来移动和缩放对应的anchor box，从而给出目标所在的位置
        '''
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / self.cell_size,
                           (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / self.cell_size,
                           np.exp(boxes[:, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 5]) / self.cell_size,
                           np.exp(boxes[:, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 5]) / self.cell_size])
        return np.transpose(boxes1, (1, 2, 3, 0))


    def calc_iou(self, box1, box2):
        # IOU计算
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def draw(self, image, result):
         # 在原图上绘制边界框
        for i in range(len(result)):
            x = int(result[i][1]) # x_center
            y = int(result[i][2]) # y_center
            w = int(result[i][3] / 2) #w/2
            h = int(result[i][4] / 2)  #h/2
            # 绘制目标边界框，矩形左上右下角
            cv2.rectangle(image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1) # 绘制矩形框存放类别名
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA #线性
            cv2.putText(
                image, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType) # 绘制文本信息(类别名和置信度)

    def image_detect(self, imagename):
        # 对图片目标进行检测
        image = cv2.imread(imagename)
        result = self.detect(image)
        self.draw(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

    def video_detect(self, cap):
        # 调用摄像头，实时检测
        while(1):
            ret, image = cap.read() #读取一帧
            if not ret:
                print('Cannot capture images from device')
                break
            result = self.detect(image)
            self.draw(image, result) # 绘制边界框 添加附加信息
            cv2.imshow('Image', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
def main():
    parser = argparse.ArgumentParser() #创建一个解析器
    parser.add_argument('--weights', default = 'yolo_v2.ckpt', type = str)    # darknet-19.ckpt
    parser.add_argument('--weight_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '', type = str)    # which gpu to be selected
    args = parser.parse_args()# 解析输入的命令行，参数默认是从sys.argv[1:]中获取，parse_args()返回一个命名空间
    # 包含传递给命令行的参数，该对象将参数保存为属性
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu    # 可见GPU
    weights_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    yolo = yolo_v2(False)    # 'False' mean 'test'
    # yolo = Darknet19(False)
    detector = Detector(yolo, weights_file)

    #detect the video
    #cap = cv2.VideoCapture('asd.mp4')
    #cap = cv2.VideoCapture(0)
    #detector.video_detect(cap)
    #detect the image
    imagename = './test/02.jpg'
    detector.image_detect(imagename)

if __name__ == '__main__':
    main()
