# -*- coding: utf-8 -*-
""" MobileNet v3的工厂函数 """
import logging

import torch
from .mobilenetv3 import MobileNetV3


__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def load_pretrained(model, model_path, load_fc):
    """ 加载预训练模型
    :param model: 模型
    :param model_path: 预训练模型文件所在路径
    :param load_fc: 是否加载前向全连接层
    """
    state_dict = torch.load(model_path)

    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('classifier.3.weight')
        state_dict.pop('classifier.3.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert all([key.find('classifier.3') != -1 for key in set(res.missing_keys)]), \
            'issue loading pretrained weights'
    del state_dict
    logging.info(f'Loaded pretrained weights for {model_path}')


def mobilenetv3_large(pretrained=False, **kwargs):
    """ 构造 MobileNetV3-Large model """
    cfgs = [
        # k,  t,  c, SE, HS, s
        [3,   1,  16, 0, 0, 1],
        [3,   4,  24, 0, 0, 2],
        [3,   3,  24, 0, 0, 1],
        [5,   3,  40, 1, 0, 2],
        [5,   3,  40, 1, 0, 1],
        [5,   3,  40, 1, 0, 1],
        [3,   6,  80, 0, 1, 2],
        [3, 2.5,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3, 2.3,  80, 0, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [3,   6, 112, 1, 1, 1],
        [5,   6, 160, 1, 1, 2],
        [5,   6, 160, 1, 1, 1],
        [5,   6, 160, 1, 1, 1]
    ]
    model = MobileNetV3(cfgs, mode='large', **kwargs)

    if pretrained:
        load_pretrained(model, 'pretrained/mobilenetv3_large.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def mobilenetv3_small(pretrained=False, **kwargs):
    """ 构造 MobileNetV3-Small模型 """

    cfgs = [
        # k,   t,  c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]
    model = MobileNetV3(cfgs, mode='small', **kwargs)

    if pretrained:
        load_pretrained(model, 'pretrained/mobilenetv3_small.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model
