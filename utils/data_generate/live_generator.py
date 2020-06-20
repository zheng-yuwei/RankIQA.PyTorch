# -*- coding: utf-8 -*-
""" 模仿数据集Live制作数据：4种失真方式 """
import os
import typing

import cv2
import numpy as np
import skimage
import glymur


class LiveDataGenerator:
    """ Live数据集噪声数据生成器 """
    # 噪声类型
    GAUSSIAN_NOISE = 'gaussian_noise'
    WHITE_NOISE = 'white_noise'
    JPEG_COMPRESS = 'jpeg_compress'
    JP2K_COMPRESS = 'jp2k_compress'
    # 噪声，值越大噪声越大
    GBLUR_LEVEL = [5, 7, 11, 15, 21]  # 高斯模糊，高斯核大小
    WN_LEVEL = [0, 3, 5, 7, 9]  # 白噪声，2^(x-10)的噪声方差
    JPEG_LEVEL = [30, 40, 60, 80, 90]  # jpeg压缩，100 - level的质量
    JP2K_LEVEL = [10, 20, 40, 60, 80]  # jpeg2000压缩倍数
    # 噪声类型和对应的噪声水平
    NOISES = {GAUSSIAN_NOISE: GBLUR_LEVEL,
              WHITE_NOISE: WN_LEVEL,
              JPEG_COMPRESS: JPEG_LEVEL,
              JP2K_COMPRESS: JP2K_LEVEL}

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def generate(self, original_dir: str):
        """ 遍历指定文件夹下的原始图像，为没张图像生成所有噪声类型、所有噪声level的噪声图像文件，及对应的标签文件
        :param original_dir: 原始图像文件夹路径
        """
        # 生成的噪声图片所在文件夹根目录
        root_dir = os.path.dirname(original_dir)
        # 遍历源目录下的图片，逐张生成噪声图片
        total_image_names = os.listdir(original_dir)
        image_pairs = []
        for indices, image_name in enumerate(total_image_names):
            if indices % 50 == 0:
                print(f'processing {indices}/{len(total_image_names)} ...')
            if not image_name.endswith(self.IMG_EXTENSIONS):
                continue
            image_path = os.path.join(original_dir, image_name)
            image = cv2.imread(image_path)
            distortion_images = self.distortion(image)
            # 将噪声图片保存到对应噪声类型的文件夹下
            for noise_type, images in distortion_images.items():
                mid_dir = os.path.join(root_dir, noise_type)
                for level, image in zip(self.NOISES[noise_type], images):
                    target_dir = os.path.join(mid_dir, noise_type + str(level))
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    target_path = os.path.join(target_dir, image_name)
                    cv2.imwrite(target_path, image)
                    image_pairs.append([image_path, target_path])

        with open(os.path.join(root_dir, 'rank.txt'), 'w+') as rank_file:
            rank_file.writelines([','.join(pair) + '\n' for pair in image_pairs])

    def distortion(self, image: np.ndarray) -> typing.Dict[str, typing.List[np.ndarray]]:
        """ 根据原始图像，生成噪声图像数据
        :param image: 原始图像
        :return 4种噪声图像：高斯噪声、白噪声、JPEG压缩噪声、JPEG2000压缩噪声
        """
        gblur_images = self.gaussian_blur(image)
        wn_images = self.white_noise(image)
        jpeg_images = self.jpeg_compress(image)
        jp2k_images = self.jp2k_compress(image)
        result = {self.GAUSSIAN_NOISE: gblur_images,
                  self.WHITE_NOISE: wn_images,
                  self.JPEG_COMPRESS: jpeg_images,
                  self.JP2K_COMPRESS: jp2k_images}
        return result

    def gaussian_blur(self, image: np.ndarray) -> typing.List[np.ndarray]:
        """ 对输入图像进行不同level的高斯模糊
        :param image: 输入图像
        :param 不同level的高斯模糊图像列表
        """
        blur_images = []
        for level in self.GBLUR_LEVEL:
            kernel = cv2.getGaussianKernel(level, level / 6.0)
            kernel = np.multiply(kernel, kernel.T)
            blur_img = cv2.filter2D(image.astype('float32'), -1, kernel,
                                    borderType=cv2.BORDER_REFLECT101)
            blur_images.append(blur_img.astype('uint8'))
        return blur_images

    def white_noise(self, image: np.ndarray) -> typing.List[np.ndarray]:
        """ 对输入图像加不同level的白噪声
        :param image: 输入图像
        :param 叠加不同level白噪声的图像列表
        """
        noisy_images = []
        for level in self.WN_LEVEL:
            noisy_img = skimage.util.random_noise(image, mode='gaussian',
                                                  seed=None, var=2 ** (10 - level))
            noisy_images.append((noisy_img * 225.).astype('uint8'))
        return noisy_images

    def jpeg_compress(self, image: np.ndarray) -> typing.List[np.ndarray]:
        """ 对输入图像进行不同level的JPEG压缩
        :param image: 输入图像
        :param 不同levelJPEG压缩的图像列表
        """
        jpeg_images = []
        for level in self.JPEG_LEVEL:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), (100 - level)]
            _, jpeg_img = cv2.imencode('.jpg', image, encode_param)
            jpeg_img = cv2.imdecode(jpeg_img, 1)
            jpeg_images.append(jpeg_img)
        return jpeg_images

    def jp2k_compress(self, image: np.ndarray) -> typing.List[np.ndarray]:
        """ 对输入图像进行不同level的JPEG2000压缩
        :param image: 输入图像
        :param 不同levelJPEG2000压缩的图像列表
        """
        jp2k_images = []
        for level in self.JP2K_LEVEL:
            jp2k_image = glymur.Jp2k('temp.jp2', data=image, cratios=[level, 1])
            jp2k_image.layer = 1
            jp2k_images.append(jp2k_image[:])
        return jp2k_images


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch RankIQA LIVE Data Generator')
    parser.add_argument('--data', default='data/train/refimgs', metavar='DIR',
                        help='原始（待失真）图像数据集路径')
    args = parser.parse_args()

    generator = LiveDataGenerator()
    generator.generate(args.data)
