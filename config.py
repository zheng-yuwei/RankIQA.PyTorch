# -*- coding: utf-8 -*-
""" 配置文件 """
import argparse


parser = argparse.ArgumentParser(description='PyTorch RankIQA Model Training')

parser.add_argument('--data', default='data', metavar='DIR', help='数据集路径')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    help='模型结构，默认：efficientnet-b0')
parser.add_argument('--image_size', default=[224, 224], type=int, nargs='+', dest='image_size',
                    help='模型输入尺寸[H, W]，默认：[224, 224]')
parser.add_argument('--num_classes', default=1, type=int, help='分支数，或者说最大分值数，默认：1')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='数据加载进程数，默认：8')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='训练batch size大小，默认：256')

# 分布式训练相关
parser.add_argument('--seed', default=None, type=int,
                    help='训练或测试时，使用随机种子保证结果的可复现，默认不使用')
parser.add_argument('--sync_bn', default=False, action='store_true',
                    help='BN同步，默认使用')
parser.add_argument('--cuda', default=True, dest='cuda', action='store_true',
                    help='是否使用cuda进行模型推理，默认 True，会根据实际机器情况调整')
parser.add_argument('-n', '--nodes', default=1, type=int, help='分布式训练的节点数')
parser.add_argument('-g', '--gpus', default=2, type=int,
                    help='每个节点使用的GPU数量，可通过设置环境变量（CUDA_VISIBLE_DEVICES=1）限制使用哪些/单个GPU')
parser.add_argument('--rank', default=-1, type=int, help='分布式训练的当前节点的序号')
parser.add_argument('--init_method', default='tcp://11.6.127.208:10006', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--logdir', default='logs', type=str, metavar='PATH',
                    help='Tensorboard日志目录，默认 logs')

# 训练过程参数设置
parser.add_argument('--train', default=False, dest='train', action='store_true',
                    help='是否训练，默认：False')
parser.add_argument('--epochs', default=85, type=int, metavar='N',
                    help='训练epoch数，默认：85')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='初始学习率，默认：1e-4', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='学习率动量')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='网络权重衰减正则项，默认: 1e-4', dest='weight_decay')
parser.add_argument('--warmup', default=5, type=int, metavar='W', help='warm-up迭代数')
parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N',
                    help='训练过程中的信息打印，每隔多少个batch打印一次，默认: 50')
parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true',
                    help='是否使用预训练模型，默认不使用')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='使用advprop的预训练模型，默认否，主要针对EfficientNet系列')

# 网络参数设置
parser.add_argument('--criterion', default='rank', type=str,
                    help='使用的损失函数，默认 rank，可选 emd，regress（打标数据集格式不一样）')
parser.add_argument('--margin', default=0.0, type=float,
                    help='margin ranking loss的margin值，默认为：0.0')

# 非训练过程参数设置
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                    help='在测试集上评估模型')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='重加载已训练好的模型 (默认: none)')
parser.add_argument('--jit', dest='jit', default=False, action='store_true',
                    help='将模型转为jit格式！')
