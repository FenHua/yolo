#!/usr/bin/python
#coding:utf-8
import os
import cv2
import numpy as np
import yolo.config as cfg
import xml.etree.ElementTree as ET

class Pascal_voc(object):
    '''获取训练数据集以及生成对应的标签文件'''
    def __init__(self):
        self.pascal_voc = os.path.join(cfg.DATA_DIR, 'Pascal_voc') # 数据文件夹
        self.image_size = cfg.IMAGE_SIZE # 图片大小
        self.batch_size = cfg.BATCH_SIZE # batch大小
        self.cell_size = cfg.CELL_SIZE # 图片划分大小
        self.classes = cfg.CLASSES # 数据集类别名称
        self.num_classes = len(self.classes) # 类别数
        self.box_per_cell = cfg.BOX_PRE_CELL #每个单元格预测box的数量
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes))) #类别->索引
        self.count = 0 # train数据集计数器，大于batch大小时置零，从新计数，同batch内避免重复
        self.epoch = 1 #存放当前训练的轮数
        self.count_t = 0 # test数据集计数器，大于batch大小时置零，从新计数，同batch内避免重复

    def load_labels(self, model):
        # 加载数据集标签
        if model == 'train':
            # 获取训练数据集文件名
            self.devkil_path = os.path.join(self.pascal_voc, 'VOCdevkit')
            self.data_path = os.path.join(self.devkil_path, 'VOC2007')
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        if model == 'test':
            # 获取测试数据集文件名
            self.devkil_path = os.path.join(self.pascal_voc, 'VOCdevkit-test')
            self.data_path = os.path.join(self.devkil_path, 'VOC2007')
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            image_ind = [x.strip() for x in f.readlines()] # 获取图片index号
        labels = [] # 存放图片路径以及标签信息
        for ind in image_ind:
            label, num = self.load_data(ind) # 获取图片的信息
            if num == 0:
                continue
            imagename = os.path.join(self.data_path, 'JPEGImages', ind + '.jpg') #图片文件路径
            labels.append({'imagename': imagename, 'labels': label}) #保存该图片的label信息
        np.random.shuffle(labels) #打乱
        return labels
    
    def load_data(self, index):
        '''
        根据index获取xml文件中每个box边界框信息
        输入图片的index；输出标签[S,S,B,5+C] 
        在VOC数据集中C=20，5+C：0:1 置信度 表示这个地方是否有目标
        1:5 目标边界框 目标中心 宽度 高度(相对与image_size)，5:25 目标类别
        '''
        label = np.zeros([self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes]) #保存图片的标签信息
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml') #获取当前index的标注文件路径
        tree = ET.parse(filename) #处理XML
        image_size = tree.find('size')
        # 得到图片宽，长信息，并计算缩放比例
        image_width = float(image_size.find('width').text)
        image_height = float(image_size.find('height').text)
        h_ratio = 1.0 * self.image_size / image_height
        w_ratio = 1.0 * self.image_size / image_width
        objects = tree.findall('object') #获取目标所在位置
        for obj in objects:
            box = obj.find('bndbox')
            # 得到目标box的中心点坐标和长宽
            x1 = max(min((float(box.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(box.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(box.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(box.find('ymax').text)) * h_ratio, self.image_size), 0)
            class_ind = self.class_to_ind[obj.find('name').text.lower().strip()] #当前类别index
            boxes = [0.5 * (x1 + x2) / self.image_size, 0.5 * (y1 + y2) / self.image_size, np.sqrt((x2 - x1) / self.image_size), np.sqrt((y2 - y1) / self.image_size)]
            # 获取目标所在单元格的下坐标
            cx = 1.0 * boxes[0] * self.cell_size
            cy = 1.0 * boxes[1] * self.cell_size
            xind = int(np.floor(cx))
            yind = int(np.floor(cy))
            label[yind, xind, :, 0] = 1 #当前单元格有目标
            label[yind, xind, :, 1:5] = boxes  #目标下坐标
            label[yind, xind, :, 5 + class_ind] = 1 #类别信息
        return label, len(objects)
    
    def next_batches(self, label):
        # 一次获取一个batch大小的图片和标签信息
        images = np.zeros([self.batch_size, self.image_size, self.image_size, 3]) #图片信息
        labels = np.zeros([self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes]) #类别信息
        num = 0
        while num < self.batch_size:
            imagename = label[self.count]['imagename'] # 获取图片路径
            images[num, :, :, :] = self.image_read(imagename) #读图
            labels[num, :, :, :, :] = label[self.count]['labels'] # 读取标签
            num += 1
            self.count += 1  # 全局变量，保证单轮不重复
             #读取完一轮数据，count 置0，当前训练轮数加1
            if self.count >= len(label):
                np.random.shuffle(label)
                self.count = 0
                self.epoch += 1
        return images, labels

    def next_batches_test(self, label):
        #  一次获取test数据集一个batch大小的图片和标签信息
        images = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
        labels = np.zeros([self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        num = 0
        while num < self.batch_size:
            imagename = label[self.count_t]['imagename']
            images[num, :, :, :] = self.image_read(imagename)
            labels[num, :, :, :, :] = label[self.count_t]['labels']
            num += 1
            self.count_t += 1
            if self.count_t >= len(label):
                self.count_t = 0
        return images, labels

    def image_read(self, imagename):
        image = cv2.imread(imagename) #读取图片
        image = cv2.resize(image, (self.image_size, self.image_size)) #缩放图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0  #[-1,1]归一化
        return image
