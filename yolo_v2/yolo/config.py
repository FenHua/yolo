#!/usr/bin/python
#coding:utf-8
# 配置文件
DATA_DIR = 'data'  #数据目录
DATA_SET = 'data_set' # 训练数据集
WEIGHTS_FILE = 'yolo_weights.ckpt' # 检查点文件

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'] # VOC数据类别名称
#ANCHOR = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHOR = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828] # 拉伸的5*2矩阵
# 通过Kmeans法对真实数据框的宽度和高度进行聚类(所有的边界框的中心点相同)，得到的5个anchor。
GPU = '' # 可见GPU设备
IMAGE_SIZE = 416    # 网络模型输入的图片大小
LEARN_RATE = 0.0001   # 学习率
MAX_ITER = 20000    # 最大迭代步
SUMMARY_ITER =10    # 每迭代10次，保存一次日志
SAVER_ITER = 50    # 每迭代50次，保存一次模型检查点
BOX_PRE_CELL = 5    # 每个划分单元cell判断5个回归框
CELL_SIZE = 13      # S值大小，即划分单元格数量
BATCH_SIZE = 10    # 训练时所选Batch的大小
ALPHA = 0.1   # 泄露函数线性激活函数系数
THRESHOLD = 0.3    # 格子是否含有目标的置信度大小
