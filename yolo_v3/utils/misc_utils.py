# coding: utf-8
import numpy as np
import tensorflow as tf
import random
from tensorflow.core.framework import summary_pb2

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])

def parse_anchors(anchor_path):
    # 解析anchor文件，返回[N, 2]形式
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors

def read_class_names(class_name_path):
    # 字典 {id:类别名}
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def shuffle_and_overwrite(file_name):
    # 打乱后重新写入
    content = open(file_name, 'r').readlines()
    random.shuffle(content)
    with open(file_name, 'w') as f:
        for line in content:
            f.write(line)

def update_dict(ori_dict, new_dict):
    # 更新字典内容
    if not ori_dict:
        return new_dict
    for key in ori_dict:
        ori_dict[key] += new_dict[key]
    return ori_dict

def list_add(ori_list, new_list):
    # 将两list相加，并返回
    for i in range(len(ori_list)):
        ori_list[i] += new_list[i]
    return ori_list

def load_weights(var_list, weights_file):
    """
    加载并转换已有的权重文件
    其中：
        var_list: 表示模型中的参数
        weights_file: 二进制文件的名字
    """
    with open(weights_file, "rb") as fp:
        # 二进制读取文件
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)
    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # 如果当前层是卷基层
        if 'Conv' in var1.name.split('/')[-2]:
            # 检查下一层
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # 加载batchnorm的参数
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                # 前移四个点，因为加载了四个参数
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # 加载偏置
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                i += 1 # 加载了一个变量
            # 加载卷基层的参数
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # 将参数改为列为主
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    return assign_ops

def config_learning_rate(args, global_step):
    # 配置学习率函数
    if args.lr_type == 'exponential':
        lr_tmp = tf.train.exponential_decay(args.learning_rate_init, global_step, args.lr_decay_freq,
                                            args.lr_decay_factor, staircase=True, name='exponential_learning_rate')
        return tf.maximum(lr_tmp, args.lr_lower_bound)
    elif args.lr_type == 'fixed':
        return tf.convert_to_tensor(args.learning_rate_init, name='fixed_learning_rate')
    else:
        raise ValueError('Unsupported learning rate type!')

def config_optimizer(optimizer_name, learning_rate, decay=0.9, momentum=0.9):
    # 配置优化器
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Unsupported optimizer type!')