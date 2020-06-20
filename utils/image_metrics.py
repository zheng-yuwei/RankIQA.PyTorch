# -*- coding: utf-8 -*-
""" 计算图像的评估指标
有参考：SROCC, PSNR, SSIM
无参考：
"""
import numpy as np


def psnr_metrics(img_1: np.ndarray, img_2: np.ndarray) -> float:
    """ 计算两个图像的PSNR值，图像为uint8的
    :param img_1: cv2读取的uint8图像
    :param img_2: cv2读取的uint8图像
    :return 两个图像的PSNR值
    """
    mse = ((img_1 - img_2).astype(np.float) ** 2).mean()
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr
