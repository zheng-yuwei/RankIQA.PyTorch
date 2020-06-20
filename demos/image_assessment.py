# -*- coding: utf-8 -*-
""" 图像质量评估模型demo """
from functools import partial

import cv2
import numpy as np
import torch
from torchvision import transforms


class ImageAssessment:
    """ 部署的质量评估模型 """
    MODEL_WEIGHT_PATH = '../checkpoints/jit_efficientnet-b0.pt'
    IMAGE_SIZE = (224, 224)

    def __init__(self):
        """ 初始化 模型、预处理器 """
        torch.set_num_threads(1)
        torch.set_flush_denormal(True)
        self.model = torch.jit.load(self.MODEL_WEIGHT_PATH)
        self.model.eval()
        self.preprocess = transforms.Compose([
            CenterCropResize(self.IMAGE_SIZE),
            # 由于预训练模型是PIL加载训练的，所以这一步要把数据转为PIL的 rgb模式
            partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def recognize(self, image) -> np.ndarray:
        """ 图像质量评估模型
        :param image: opencv bgr格式的numpy数组
        :return 每一分支的概率
        """
        image = self.preprocess(image)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))[0]
        probabilities = torch.sigmoid(output).detach().numpy()
        return probabilities


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


if __name__ == '__main__':
    import os
    recognizer = ImageAssessment()
    root_dir = 'images'
    for image_name in os.listdir(root_dir):
        image_path = os.path.join(root_dir, image_name)
        img = cv2.imread(image_path)
        result = recognizer.recognize(img)
        print(f'{image_name}: {result}')
