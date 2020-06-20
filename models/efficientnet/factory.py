# -*- coding: utf-8 -*-
""" EfficientNet系列的工厂脚本 """
from .model import EfficientNet

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
           'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
           'efficientnet_b8']


def load_efficientnet(name, num_classes, adv_prop, pretrained):
    if pretrained:
        model = EfficientNet.from_pretrained(name,
                                             num_classes=num_classes,
                                             adv_prop=adv_prop)
    else:
        model = EfficientNet.from_name(name, num_classes=num_classes)
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b0', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b1(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b1', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b2(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b2', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b3(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b3', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b4(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b4', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b5(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b5', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b6(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b6', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b7(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b7', num_classes, adv_prop, pretrained)
    return model


def efficientnet_b8(pretrained=False, **kwargs):
    num_classes = kwargs.get('num_classes') or 1000
    adv_prop = kwargs.get('adv_prop') or False
    model = load_efficientnet('efficientnet-b8', num_classes, adv_prop, pretrained)
    return model
