#!/usr/bin/python
#coding:utf-8
import os
# 数据集参数设置
DATA_PATH = 'data'  # 数据集目录
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')  # VOC数据集
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')       # 保存生成的数据集标签缓冲文件
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')      # 保存生成网络模型和日志文件
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')   #  模型参数保存(检查点文件)
# WEIGHTS_FILE = None   # 是否进行模型恢复
WEIGHTS_FILE ='./data/pascal_voc/weights/YOLO_small.ckpt'
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor'] #VOC 数据类别名
FLIPPED = True # 使用水平镜像(扩充数据集)

# 模型参数设置
IMAGE_SIZE = 448 # 输入图片大小
CELL_SIZE = 7        # 单元格大小
BOXES_PER_CELL = 2 # 每个单元格预测盒子的数量
ALPHA = 0.1 # 泄露修正线性激活函数系数
DISP_CONSOLE = False # 控制台输出信息
OBJECT_SCALE = 1.0       # loss函数中有目标时的置信度权重 
NOOBJECT_SCALE = 1.0  # loss函数中没有目标时的置信度权重
CLASS_SCALE = 2.0          # loss函数中类别判断函数的权重
COORD_SCALE = 5.0        # loss函数中边界框优化函数的权重

# 求解器参数设置
GPU = ''
LEARNING_RATE = 0.0001  # 学习率
DECAY_STEPS = 30000        # 退化学习率衰减步数
DECAY_RATE = 0.1              # 衰减率
STAIRCASE = True 
BATCH_SIZE = 10                 # 批量大小
MAX_ITER = 15000               # 最大迭代次数
SUMMARY_ITER = 10           # 日志文件保存间隔步
SAVE_ITER = 1000                # 模型保存间隔步

# 测试参数设置
THRESHOLD = 0.2          # 格子是否有目标的置信度阈值
IOU_THRESHOLD = 0.5  # NMS，IOU阈值
