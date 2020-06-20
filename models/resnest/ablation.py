# -*- coding: utf-8 -*-
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Created by: Hang Zhang
#  Email: zhanghang0704@gmail.com
#  Copyright (c) 2020
#
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNeSt ablation study models"""
from .resnet import ResNet, Bottleneck, load_pretrained

__all__ = ['resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d', 'resnest50_fast_4s1x64d',
           'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']


def resnest50_fast_1s1x64d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_1s1x64d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest50_fast_2s1x64d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_2s1x64d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest50_fast_4s1x64d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_4s1x64d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest50_fast_1s2x40d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_1s2x40d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest50_fast_2s2x40d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_2s2x40d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest50_fast_4s2x40d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=4, groups=2, bottleneck_width=40,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_4s2x40d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model


def resnest50_fast_1s4x24d(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=1, groups=4, bottleneck_width=24,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        load_pretrained(model, 'pretrained/resnest50_fast_1s4x24d.pth',
                        load_fc=(kwargs.get('num_classes') in (1000, None)))
    return model
