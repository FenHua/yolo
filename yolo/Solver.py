#!/usr/bin/python
#coding:utf-8
import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim

class Solver(object):
# 求解器，用于训练yolo
    def __init__(self, net, data):
        #构造函数，加载训练参数
        self.net = net      # YOLONET对象
        self.data = data  #  voc数据
        self.weights_file = cfg.WEIGHTS_FILE # 检查点文件路径
        self.max_iter = cfg.MAX_ITER  # 训练最大迭代次数
        self.initial_learning_rate = cfg.LEARNING_RATE # 初始学习率
        self.decay_steps = cfg.DECAY_STEPS # 衰减步数
        self.decay_rate = cfg.DECAY_RATE    # 衰减率
        self.staircase = cfg.STAIRCASE            # staircase 楼梯 bool型
        self.summary_iter = cfg.SUMMARY_ITER # 日志保存间隔步
        self.save_iter = cfg.SAVE_ITER  # 模型保存间隔步
        # 输出文件夹路径
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg() # 保存配置信息
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None) # 保存所有变量
        self.ckpt_file = os.path.join(self.output_dir, 'yolo') # 指定保存模型名称
        self.summary_op = tf.summary.merge_all()  # 合并所有的日子文件
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60) # 指定文件路径，用于写日志
        self.global_step = tf.train.create_global_step() # 当前迭代次数
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate') # 退化学习率
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step) # 执行操作
        gpu_options = tf.GPUOptions() # 设置GPU资源
        config = tf.ConfigProto(gpu_options=gpu_options) # 按需分配GPU使用的资源
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer()) #初始化变量
        if self.weights_file is not None:
            # 从现有的模型恢复
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)
        self.writer.add_graph(self.sess.graph) # 将图写入日志文件
        
    def train(self):
        # 训练函数
        train_timer = Timer()
        load_timer = Timer()
        # 开始迭代
        for step in range(1, self.max_iter + 1):
            load_timer.tic() # 数据加载起始时间
            images, labels = self.data.get() #获取数据和标签
            load_timer.toc() # 数据加载所耗时间
            feed_dict = {self.net.images: images, self.net.labels: labels}
            if step % self.summary_iter == 0:
                #每迭代summary_iter次保存一次日志文件
                if step % (self.summary_iter * 10) == 0:
                    #每迭代10×summary_iter输出一次日志
                    train_timer.tic()  # 每次训练的开始时间
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc() # 每次训练的截止时间
                    # 输出信息
                    log_str = '{} Epoch: {}, Step: {}, Learning rate: {},Loss: {:5.3f}\nSpeed: {:.3f}s/iter,Load: {:.3f}s/iter, Remain: {}'.format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)
                else:
                    train_timer.tic()     # 每次迭代训练的起始时间
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()   # 每次迭代训练的终止时间
                self.writer.add_summary(summary_str, step)  #将summary写入文件
            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict) #每次迭代global_step++
                train_timer.toc()
            if step % self.save_iter == 0:
                # 保存模型
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):
        # 保存配置信息
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

def update_config_paths(data_dir, weights_file):
    # 数据集路径和模型检查点文件路径
    cfg.DATA_PATH = data_dir   #数据所在文件夹
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc') #voc数据集所在文件夹
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')      #  保存生成的数据集标签文件夹
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')     #  生成网络模型和日志文件所在的文件夹
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')  #  检查点文件目录
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)