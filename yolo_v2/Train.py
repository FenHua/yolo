# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import datetime
import time
import os
import yolo.config as cfg

from pascal_voc import Pascal_voc
from six.moves import xrange
from yolo.yolo_v2 import yolo_v2
# from yolo.darknet19 import Darknet19

class Train(object):
    def __init__(self, yolo, data):
        # 构造函数
        self.yolo = yolo # 网络模型 
        self.data = data # 数据集
        self.num_class = len(cfg.CLASSES) #类别数
        self.max_step = cfg.MAX_ITER #最大迭代次数
        self.saver_iter = cfg.SAVER_ITER #网络模型保存间隔步
        self.summary_iter = cfg.SUMMARY_ITER # 日志保存间隔步
        self.initial_learn_rate = cfg.LEARN_RATE #学习率
        self.output_dir = os.path.join(cfg.DATA_DIR, 'output') #输出文件夹路径
        weight_file = os.path.join(self.output_dir, cfg.WEIGHTS_FILE) #检查点文件路径
        self.variable_to_restore = tf.global_variables()  
        self.saver = tf.train.Saver(self.variable_to_restore) #保存所有变量
        self.summary_op = tf.summary.merge_all() #合并所有的日志文件
        self.writer = tf.summary.FileWriter(self.output_dir) #指定文件路径，用于写日志
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False) # 当前迭代次数
        self.learn_rate = tf.train.exponential_decay(self.initial_learn_rate, self.global_step, 20000, 0.1, name='learn_rate') #退化学习率
        # self.global_step = tf.Variable(0, trainable = False)
        # self.learn_rate = tf.train.piecewise_constant(self.global_step, [100, 190, 10000, 15500], [1e-3, 5e-3, 1e-2, 1e-3, 1e-4])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.yolo.total_loss, global_step=self.global_step) #优化
        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.average_op)

        config = tf.ConfigProto(gpu_options=tf.GPUOptions()) # 设置GPU资源
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer()) # 初始化变量
        print('Restore weights from:', weight_file)
        self.saver.restore(self.sess, weight_file) #恢复模型
        self.writer.add_graph(self.sess.graph) # 将模型写入日志文件

    def train(self):
        # 得到label信息
        labels_train = self.data.load_labels('train')
        labels_test = self.data.load_labels('test')
        num = 5
        initial_time = time.time() # 训练开始时间
        for step in xrange(0, self.max_step + 1):
            #迭代
            images, labels = self.data.next_batches(labels_train) # 获取一个batch的数据
            feed_dict = {self.yolo.images: images, self.yolo.labels: labels} #tensorflow输入字典
            if step % self.summary_iter == 0:
                # 每迭代summary_iter次保存一次日志文件
                if step % 50 == 0:
                    # 输出一次日志
                    summary_, loss, _ = self.sess.run([self.summary_op, self.yolo.total_loss, self.train_op], feed_dict = feed_dict)
                    sum_loss = 0
                    for i in range(num):
                        # 从测试数据集中取5个batch的数据，进行测试网络模型
                        images_t, labels_t = self.data.next_batches_test(labels_test)
                        feed_dict_t = {self.yolo.images: images_t, self.yolo.labels: labels_t}
                        loss_t = self.sess.run(self.yolo.total_loss, feed_dict=feed_dict_t)
                        sum_loss += loss_t
                    log_str = ('{} Epoch: {}, Step: {}, train_Loss: {:.4f}, test_Loss: {:.4f}, Remain: {}').format(
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.data.epoch, int(step), loss, sum_loss/num, self.remain(step, initial_time))
                    print(log_str)
                    if loss < 1e4:
                        pass
                    else:
                        print('loss > 1e04')
                        break  # 训练终止
                else:
                    summary_, _ = self.sess.run([self.summary_op, self.train_op], feed_dict = feed_dict)
                self.writer.add_summary(summary_, step) #写日志
            else:
                self.sess.run(self.train_op, feed_dict = feed_dict)
            if step % self.saver_iter == 0:
                self.saver.save(self.sess, self.output_dir + '/yolo_v2.ckpt', global_step = step) #保存网络模型
    def remain(self, i, start):
        # 返回训练还需时间
        if i == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - i) / i
        return str(datetime.timedelta(seconds = int(remain_time)))
