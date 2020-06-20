# -*- coding: utf-8 -*-
""" 主程序main.py调用的应用
convert: 模型转换
train: 基于训练集、验证集的模型训练
test: 基于测试集的模型测试
predict: 基于图片文件夹的模型预测
"""
from .convert import convert_to_jit
from .test import test
from .train import train
