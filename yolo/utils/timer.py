#!/usr/bin/python
#coding:utf-8
import time
import datetime
class Timer(object):
    #计时器
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        self.start_time = time.time() # 使用time.time，time.clock 处理多线程不佳

    def toc(self, average=True):
        # 终止时间
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        # 根据先前的迭代次数和迭代时间，根据剩余的迭代次数，估计剩余迭代时间
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))
