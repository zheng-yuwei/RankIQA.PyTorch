# -*- coding: utf-8 -*-
""" PyTorch官方提供的预定义模型及自定义模型 """
import logging

import torch
from torchvision import models
from .efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_b8
)
from .mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from .resnest import (
    resnest50, resnest101, resnest200, resnest269,
    resnest50_fast_1s1x64d, resnest50_fast_2s1x64d, resnest50_fast_4s1x64d,
    resnest50_fast_1s2x40d, resnest50_fast_2s2x40d, resnest50_fast_4s2x40d,
    resnest50_fast_1s4x24d
)


models_map = {
    'efficientnet_b0': efficientnet_b0, 'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2, 'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4, 'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6, 'efficientnet_b7': efficientnet_b7,
    'efficientnet_b8': efficientnet_b8,

    'mobilenetv3_small': mobilenetv3_small, 'mobilenetv3_large': mobilenetv3_large,

    'resnest50': resnest50, 'resnest101': resnest101, 'resnest200': resnest200, 'resnest269': resnest269,
    'resnest50_fast_1s1x64d': resnest50_fast_1s1x64d, 'resnest50_fast_2s1x64d': resnest50_fast_2s1x64d,
    'resnest50_fast_4s1x64d': resnest50_fast_4s1x64d, 'resnest50_fast_1s2x40d': resnest50_fast_1s2x40d,
    'resnest50_fast_2s2x40d': resnest50_fast_2s2x40d, 'resnest50_fast_4s2x40d': resnest50_fast_4s2x40d,
    'resnest50_fast_1s4x24d': resnest50_fast_1s4x24d,
}


def get_model(name, pretrained=False, **kwargs):
    """ 获取指定名称的模型
    :param name: 指定模型名称
    :param pretrained: 是否加载预训练模型
    :param kwargs: num_classes等
    :return 指定名称的模型
    """
    if name in models_map:
        model = models_map[name](pretrained, **kwargs)
    else:
        model = models.__dict__[name](**kwargs)
        if pretrained:
            model_path = f'pretrained/{name}.pth'
            state_dict = torch.load(model_path)
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            acc = model.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(acc.missing_keys) == {'fc.weight', 'fc.bias'}, 'issue loading pretrained weights'
            logging.info(f"=> using pre-trained model '{model_path}'")

    return model
