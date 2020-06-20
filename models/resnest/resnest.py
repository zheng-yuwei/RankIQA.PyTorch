# -*- coding: utf-8 -*-
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Created by: Hang Zhang
#  Email: zhanghang0704@gmail.com
#  Copyright (c) 2020
#
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt models"""
from .resnet import ResNet, Bottleneck, load_pretrained

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269', ]


def resnest50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest101.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest200(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest200.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest269(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest269.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model
