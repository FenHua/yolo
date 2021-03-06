#!/usr/bin/python
#coding:utf-8
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg

class pascal_voc(object):
    '''获取训练数据集以及生成对应的标签文件'''
    def __init__(self, phase, rebuild=False):
        '''phase：train或者test；rebuild表示是否重新创建标签文件，保存在缓存区文件夹'''
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit') # VOCdevkit文件夹路径
        self.data_path = os.path.join(self.devkil_path, 'VOC2012') # VOC2012文件夹路径
        self.cache_path = cfg.CACHE_PATH # cache文件所在的文件夹
        self.batch_size = cfg.BATCH_SIZE # 批大小
        self.image_size = cfg.IMAGE_SIZE # 图像大小
        self.cell_size = cfg.CELL_SIZE # 单元格大小
        self.classes = cfg.CLASSES # 数据集类别名称
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes)))) #类别名->索引
        self.flipped = cfg.FLIPPED #水平镜像扩充
        self.phase = phase # 训练or测试
        self.rebuild = rebuild # 是否重新创建数据集标签文件
        self.cursor = 0  #gt_labels加载数据，cursor表明当前读取到第几个
        self.epoch = 1 # 存放当前训练的轮数
        self.gt_labels = None
        self.prepare() #加载数据集标签，初始化gt_labels

    def get(self):
        '''
        从gt_labels集合随机读取batch大小的图片和对应的标签
        return:
        images:[batch,448,448,3] labels:[batch,7,7,25]
        '''
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25)) # 坐标信息，默认置信度以及类别信息
        count = 0 # 计数器
        while count < self.batch_size:
            # 获取batch个数据
            imname = self.gt_labels[self.cursor]['imname'] # 获取图片路径
            flipped = self.gt_labels[self.cursor]['flipped'] # 是否使用水平镜像
            images[count, :, :, :] = self.image_read(imname, flipped) #读图
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']  # 读取标签
            count += 1
            self.cursor += 1 # 全局变量，保证单轮不重复
            # 读取完一轮数据，cursor置0，当前训练轮数加1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname) #读取图片
        image = cv2.resize(image, (self.image_size, self.image_size)) #缩放图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0 #[-1,1]归一化
        if flipped:
            image = image[:, ::-1, :] #水平镜像
        return image

    def prepare(self):
        '''初始化数据集的标签，保存在变量gt_labels中'''
        gt_labels = self.load_labels() # 一个复杂的dict，图片所对应的各种信息
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels) #深拷贝
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True 
                gt_labels_cp[idx]['label'] =gt_labels_cp[idx]['label'][:, ::-1, :] #所有目标所在的格子都进行水平镜像
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        # 置信度等于1，表明格子有目标
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1] #中心x坐标水平镜像，有坐标不变
            gt_labels += gt_labels_cp # 将水平镜像数据标签追加数据集标签后面
        np.random.shuffle(gt_labels) # 打乱数据
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        '''
        加载数据集标签
        返回gt_labels，是一个复杂的dict，对应一张图片
        imname 图片路径，label 图片对应的标签[7,7,25]的矩阵 flipped 是否使用水平镜像
        '''
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl') # 缓冲文件名：用来保存数据集标签的文件
        if os.path.isfile(cache_file) and not self.rebuild:
            #文件存在，不重新创建，直接读取
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels
        print('Processing gt_labels from: ' + self.data_path)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        #获取训练测试集的数据集文件名
        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
        gt_labels = [] # 存放图片的标签，图片路径，是否使用镜像
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index) # 获取每张图片信息
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg') # 图片文件路径
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False}) # 保存该图片的信息
        print('Saving gt_labels to: ' + cache_file)
        # 保存
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        从XML文件中获取每个box的边界信息
        输入：图片文件的index；输出：标签[7, 7, 25]
        0:1 置信度 表示这个地方是否有目标
        1:5 目标边界框 目标中心 宽度 高度(实际值，没有归一化)
        5:25 目标类别
        len(objs)：objs对象长度
        """
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg') #获取图片文件名路径
        im = cv2.imread(imname)
        # 长宽缩放比例
        h_ratio = 1.0 * self.image_size / im.shape[0] 
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])
        label = np.zeros((self.cell_size, self.cell_size, 25)) # 用于保存图片文件的标签
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename) # 处理xml文件
        objs = tree.findall('object')
        for obj in objs:
            bbox = obj.find('bndbox')
            # 图片缩放操作，坐标从0开始
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()] # 分类名->类别index
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]# box信息[中心点，宽，高]
            # 格子下坐标的计算
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            if label[y_ind, x_ind, 0] == 1:
                continue #当前图片已经初始化
            label[y_ind, x_ind, 0] = 1 #当前box有目标
            label[y_ind, x_ind, 1:5] = boxes #目标下坐标（相对于448原始图而言）
            label[y_ind, x_ind, 5 + cls_ind] = 1 #目标类别信息(独热)
        return label, len(objs)
