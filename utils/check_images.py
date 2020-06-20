# -*- coding: utf-8 -*-
"""Description"""
import logging

import os
from PIL import Image


def check(image_paths):
    """ 检查图像文件是否可读、可用有效
    :param image_paths: 图像文件所在路径
    """
    bad_paths = list()
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            img.resize((360, 640))
        except Exception:
            bad_paths.append(image_path)
    return bad_paths


def check_set(set_dir):
    """ 检查整个图像数据集中的图像文件是否可读有效
    :param set_dir: 图像数据集文件夹路径
    """
    for mid_dir in os.listdir(set_dir):
        image_dir = os.path.join(set_dir, mid_dir)
        image_paths = [os.path.join(image_dir, image_name)
                       for image_name in os.listdir(image_dir)]
        logging.info(check(image_paths))


if __name__ == '__main__':
    for _set_dir in ['train', 'test', 'val']:
        check_set(_set_dir)
