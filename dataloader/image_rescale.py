# -*- coding: utf-8 -*-
""" 将图片重新等比例伸缩到指定尺寸 """
import logging

import cv2
import numpy as np
from torch import nn
from scipy.stats import truncnorm


class CenterCropResize:
    """ 将图片中间区域crop出来，并resize到目标尺寸
    参数:
        output_size (init 或者 tuple): 要求的输出尺寸. 如果是tuple, 输出和output_size匹配。
        如果是int, 图片的边长都为 output_size。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.h_w_ratio = 1.0 * self.output_size[0] / self.output_size[1]

    def __call__(self, image):
        crop_image = self.center_crop(image, self.h_w_ratio)
        image = cv2.resize(crop_image, self.output_size[::-1])
        return image

    @staticmethod
    def center_crop(image, h_w_ratio=1):
        """ 将图像按高宽比，将中间区域裁剪出来
        :param image: 原图
        :param h_w_ratio: 宽高比
        :return: 裁剪出来的图像
        """
        h, w = image.shape[:2]
        crop_h = int(w * h_w_ratio)
        if crop_h < h:
            h_start = (h - crop_h) // 2
            h_end = h_start + crop_h
            crop_image = image[h_start:h_end, :, :]
        else:
            crop_w = int(h / h_w_ratio)
            w_start = (w - crop_w) // 2
            w_end = w_start + crop_w
            crop_image = image[:, w_start:w_end, :]
        return crop_image


class MultiScale:
    """将样本中图片修改为规定的尺寸. 尺寸是根据正态分布随机生成的
    参数:
        output_size (init 或者 tuple): 要求的输出尺寸. 如果是tuple, 输出和output_size匹配。
        如果是int, 图片的短边是output_size，长边按比例缩放。
    """

    def __init__(self, output_size, h_w_ratio=1.8):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, (tuple, list)):
            h_w_ratio = 1.0 * output_size[0] / output_size[1]
            output_size = output_size[1]
        logging.info(f'>>>>>>> Use H:W ratio {h_w_ratio}, base width {output_size}')
        self.h_w_ratio = h_w_ratio
        # 2 * sigma = ratio_95 * output_size, 亦即(1 + [-ratio_95, ratio_95]) * output_size 里包含95.45%
        ratio_95 = 0.15
        mean = output_size
        norm_delta = 2.0
        sigma = ratio_95 * output_size / norm_delta
        self.generator = truncnorm(-norm_delta, norm_delta, loc=mean, scale=sigma)
        # plt.hist(self.generator.rvs(1000))

    def __call__(self, images):
        target_w = int(np.round(self.generator.rvs()))
        target_h = int(np.round(target_w * self.h_w_ratio))
        crop_images = self.center_crop(images, self.h_w_ratio)
        # 等比例缩放 images.permute((0, 3, 1, 2))
        images = nn.functional.interpolate(crop_images, size=(target_h, target_w),
                                           mode='bilinear', align_corners=False)
        return images

    @staticmethod
    def center_crop(images, h_w_ratio=1):
        """ 将图像按高宽比，将中间区域裁剪出来
        :param images: 原图, NCHW
        :param h_w_ratio: 宽高比
        :return: 裁剪出来的图像
        """
        h, w = images.shape[2:]
        crop_h = int(w * h_w_ratio)
        if crop_h < h:
            h_start = (h - crop_h) // 2
            h_end = h_start + crop_h
            crop_images = images[:, :, h_start:h_end, :]
        else:
            crop_w = int(h / h_w_ratio)
            w_start = (w - crop_w) // 2
            w_end = w_start + crop_w
            crop_images = images[..., w_start:w_end]
        return crop_images
