# -*- coding: utf-8 -*-
""" EfficientNet工具 """
import os
import math
import logging

import torch

from .config import GlobalParams


def round_filters(filters: int, global_params: GlobalParams) -> int:
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, global_params: GlobalParams) -> int:
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs: torch.Tensor, p: float,
                 training: bool) -> torch.Tensor:
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def load_pretrained_weights(model, model_name, load_fc=True, adv_prop=False):
    """ 加载预训练模型
    :param model: 模型
    :param model_name: 模型名称
    :param load_fc: 是否复用fc层
    :param adv_prop: 是否使用adv_prop预训练模型
    """
    model_root = 'pretrained'
    if adv_prop:
        model_root = os.path.join(model_root, 'advprop')
    model_path = os.path.join(model_root, f'{model_name}.pth')
    state_dict = torch.load(model_path)

    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert all([key.find('_fc') != -1 for key in set(res.missing_keys)]), 'issue loading pretrained weights'
    del state_dict
    logging.info(f'Loaded pretrained weights for {model_path}')
