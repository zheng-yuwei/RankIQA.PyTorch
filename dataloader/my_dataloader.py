# -*- coding: utf-8 -*-
""" 数据集加载 """
import os
import typing
import logging
from functools import partial

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

from dataloader.image_rescale import CenterCropResize


def load(args, name='train'):
    """ 数据集加载
    :param args: 训练参数
    :param name: 加载的数据集类型，train,test,val
    """
    names = ('train', 'test', 'val')
    assert name in names, f'Name of dataset must in {names}!'

    data_dir = os.path.join(args.data, name)
    dataset = RankImageFolder(data_dir, args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    loader = DataLoaderX(dataset, batch_size=args.batch_size,
                         shuffle=(name == 'train' and not args.distributed),
                         num_workers=args.workers, pin_memory=True,
                         collate_fn=DataLoaderX.my_collate_fn,
                         sampler=train_sampler, drop_last=args.distributed)

    return loader


class DataLoaderX(DataLoader):
    """ 使用prefetch_generator包提供的数据预加载功能 """
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    @staticmethod
    def my_collate_fn(batch):
        """ 将图像对、DMOS分值对、路径对的列表，进行数据整理为图像列表和路径列表
        :param batch: 图像对、路径对的列表，[[[image_a0, image_a1], [score_a0, score_a1], [path_a0, path_a1]], ...]
        :return 图像列表和路径列表，
                [image_a0, image_a1, image_b0, image_b1, ...]
                [score_a0, score_a1, score_b0, score_b1, ...]
                [path_a0, path_a1, path_b0, path_b1, ...]
        """
        image_sequence = []
        score_sequence = []
        path_sequence = []
        for image_pair, score_pair, path_pair in batch:
            image_sequence.extend(image_pair)
            score_sequence.extend(score_pair)
            path_sequence.extend(path_pair)

        # 列表的Tensor堆成一个Tensor，列表成为新维
        image_sequence = torch.utils.data.dataloader.default_collate(image_sequence)
        score_sequence = torch.utils.data.dataloader.default_collate(score_sequence)
        path_sequence = torch.utils.data.dataloader.default_collate(path_sequence)
        return image_sequence, score_sequence, path_sequence


class RankImageFolder(Dataset):
    """ 排序图片文件夹Dataset类 """

    def __init__(self, root, args):
        """ 排序图片文件夹Dataset类
        :param root: 排序图像所在文件夹，该文件夹下有 rank.txt 标签文件，每行格式如下：
                     good_image_path.jpg,bad_image_path.jpg
                     图片格式不一定为jpg格式，以上表示左边图片质量 > 右边图片质量
        :param args: 图像加载器的入参，图像预处理尺寸等超参
        """
        file_name = 'label.txt'
        self.label_lines = list()
        with open(os.path.join(root, file_name)) as label_file:
            for line in label_file:
                self.label_lines.append([p.strip() for p in line.strip().split(',')])

        # 图片加载器，包含cv2读取和图像预处理
        self.loader = self.get_loader(args)
        self.is_rank = (args.criterion == 'rank')
        self.is_regress = (args.criterion == 'regress')
        self.is_emd = (args.criterion == 'emd')

    def __getitem__(self, index: int) -> (typing.Tuple[np.ndarray, np.ndarray],
                                          typing.Tuple[str, str],
                                          typing.Tuple[float, float]):
        """ 读取数据，得到 图像，分数，文件路径
        :param index: Index
        :return tuple: (image_pair, score_pair, path_pair) 一对图像，对应的DMOS分数，对应的路径
        """
        label_line = self.label_lines[index]
        if self.is_rank:
            path_pair = label_line
            image_pair = [self.loader(image_path) for image_path in label_line]
            score_pair = [-1, -1]
        elif self.is_regress:
            path_pair = label_line[0:1]
            image_pair = [self.loader(label_line[0])]
            score_pair = [np.array(label_line[1], dtype=np.float32)]
        elif self.is_emd:
            path_pair = label_line[0:1]
            image_pair = [self.loader(label_line[0])]
            score_pair = [np.array([float(num) for num in label_line[1:]], dtype=np.float32)]
        else:
            logging.error("No Data set Reader Implemented for your Criterion, "
                          "only (rank, regress, emd) allow!")
            raise NotImplementedError('No Data set Reader Implemented for your Criterion')
        return image_pair, score_pair, path_pair

    def __len__(self):
        return len(self.label_lines)

    @staticmethod
    def get_loader(args):
        """ 图像加载及预处理器
        :param args: 训练/测试等的参数
        """
        logging.info(f'Using image size: {args.image_size}')
        if args.advprop:
            normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        transforms_list = [
            cv2.imread,
            CenterCropResize(args.image_size),
            # 由于预训练模型是PIL加载训练的，所以这一步要把数据转为PIL的 rgb模式
            partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB),
            transforms.ToTensor(),
            normalize,
        ]
        return transforms.Compose(transforms_list)


if __name__ == '__main__':
    import argparse
    my_args = argparse.Namespace()
    my_args.data = 'data'
    my_args.batch_size = 3
    my_args.workers = 0
    my_args.image_size = [224, 224]
    my_args.advprop = False
    my_args.criterion = 'rank'

    data_loader = load(my_args, name='train')
    for my_samples, my_scores, image_paths in data_loader:
        print(image_paths)
        print(my_scores)
        # 解归一化
        my_samples = (my_samples * np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)) +
                      np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))) * 255
        my_samples = my_samples.numpy().transpose(0, 2, 3, 1).astype(np.uint8)[..., ::-1]
        for my_image in my_samples:
            cv2.imshow('a', my_image)
            cv2.waitKey(0)
