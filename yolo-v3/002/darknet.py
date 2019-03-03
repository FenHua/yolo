#!/usr/bin/python
#coding:utf-8
import torch
import torch.nn as nn
from util import predict_transform
import numpy as np
'''
配置文件定义了6种不同type
'net': 相当于超参数,网络全局配置的相关参数
{'convolutional', 'net', 'route', 'shortcut', 'upsample', 'yolo'}
'''
def parse_cfg(cfgfile):
    # 解析配置文件
    # 输入: 配置文件路径；返回值: list对象,其中每一个item为一个dict类型
    # 对应于一个要建立的神经网络模块
    # 加载文件并过滤掉文本中多余内容
    with open(cfgfile, 'r') as f:
        lines = f.read().split('\n') # 以换行符切分
        lines = [x for x in lines if len(x) > 0]  # 去掉空行
        lines = [x for x in lines if x[0]!='#']    # 去掉已经注释的行
        lines = [x.rstrip().lstrip() for x in lines]  # 去掉每一行左右两边的空格(strip 剥夺)
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            # 判断是否是某个层的开始
            if len(block) != 0:
                # 当前块内已经存放了上一个块
                blocks.append(block)
                block = {}  # 新建一个空白块存描述信息
            block["type"] = line[1:-1].rstrip()  # 块名
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()  # 记录配置信息
    blocks.append(block)  # 退出循环，将最后一个未加入的block加进去
    return blocks

class EmptyLayer(nn.Module):
   # 为shortcut layer / route layer 准备, 具体功能不在此实现
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    # yolo 检测层
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
    def forward(self, x, input_dim, num_classes, confidence):
        x = x.data
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, input_dim, self.anchors, num_classes, confidence, CUDA)
        return prediction

def create_modules(blocks):
    net_info = blocks[0] #获取模型参数信息
    module_list = nn.ModuleList() # nn pytorch网络
    index = 0 # route layer 会用到
    previous_filters = 3 # 初始值对应于输入数据3通道
    output_filters = []
    # 构建网络
    for block in blocks:
        container = nn.Sequential() # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        if block["type"] == "net":
            continue # 过滤掉网络头
        if block["type"] == "convolutional":
            # 1. 卷积层 
            activation = block["activation"] # 获取激活函数
            try:
                batch_normalize = int(block["batch_normalize"]) # 批归一化
                bias = False
            except:
                batch_normalize = 0
                bias = True
            # 卷积层参数
            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            if padding:
                pad = (kernel_size - 1) // 2  # "// " 表示整数除法
            else:
                pad = 0
            conv = nn.Conv2d(previous_filters, filters, kernel_size, stride, pad, bias=bias) 
            container.add_module("conv_{0}".format(index), conv) # 开始创建并添加相应层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                container.add_module("batch_norm_{0}".format(index), bn) # 添加batch_normalization
            if activation == "leaky":
                # 判断激活函数类型(默认是线性的)
                activn = nn.LeakyReLU(0.1, inplace=True) # 给定参数负轴系数0.1
                container.add_module("leaky_{0}".format(index), activn)

        elif block["type"] == "upsample":
            # 上采样层，没有使用 Bilinear2dUpsampling，使用的为最近邻插值
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            container.add_module("upsample_{}".format(index), upsample)

        elif block["type"] == "route":
            # 好几个层的输出转接到一起作为本层的输出
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0]) #一个route的开始
            try:
                end = int(block["layers"][1]) #尾点
            except:
                end = 0
            #Positive anotation: 正值
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            container.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
        
        elif block["type"] == "shortcut":
            # 完成相应层的相加
            from_ = int(block["from"])
            shortcut = EmptyLayer()
            container.add_module("shortcut_{}".format(index), shortcut)

        elif block["type"] == "maxpool":
            # 池化层
            stride = int(block["stride"])
            size = int(block["size"])
            maxpool = nn.MaxPool2d(size, stride)
            container.add_module("maxpool_{}".format(index), maxpool)

        elif block["type"] == "yolo":
            # Yolo是检测层
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask] # 实验中所使用的anchor
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors) # 锚点,检测,位置回归,分类
            container.add_module("Detection_{}".format(index), detection)
        else:
            assert False
        module_list.append(container)
        previous_filters = filters
        output_filters.append(filters)
        index += 1
    return net_info, module_list

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()    # super() 函数是用于调用父类(超类)的一个方法
        self.blocks = parse_cfg(cfgfile)    # 得到配置信息
        self.net_info, self.module_list = create_modules(self.blocks) # 建模型和获取网络信息
        self.header = torch.IntTensor([0, 0, 0, 0]) # 模型版本标志
        self.seen = 0

    def get_blocks(self):
        return self.blocks # list

    def get_module_list(self):
        return self.module_list # nn.ModuleList

    def forward(self, x, CUDA=True):
        detections = []
        modules = self.blocks[1:]  # 除了net块之外的所有
        outputs = {} # cache output for route layer
        write = False # 拼接检测层结果
        for i in range(len(modules)):
            module_type = modules[i]["type"]
            if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x #
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                outputs[i] = x
            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i - 1] + outputs[i + from_]  # 求和运算
                outputs[i] = x
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"]) # 得到输入维度
                num_classes = int(modules[i]["classes"]) # 得到类别数
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA) # 输出结果
                if type(x) == int:
                    continue
                if not write:
                    # 将在3个不同level的feature map上检测结果存储在 detections 里
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i - 1]
        try:
            return detections   # 网络forward 执行完毕
        except:
            return 0

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb") # 读取权值参数文件
        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        weights = np.fromfile(fp, dtype=np.float32) # 网络模型参数
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    num_bn_biases = bn.bias.numel() #得到batchnormalization的个数
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases]) #加载权重
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    # 将加载的权值送入网络模型
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel() # 偏置项数量
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]) #加载卷积层的偏置权值
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data) #根据模型改变偏置呈现样式
                    conv.bias.data.copy_(conv_biases) #将数据复制到模型中
                num_weights = conv.weight.numel() # 卷积层数
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                # 循环完成赋值
