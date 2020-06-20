# -*- coding: utf-8 -*-
""" 评估量：记录，打印 """
import logging


class AverageMeter:
    """ 计算并存储 评估量的均值和当前值 """
    def __init__(self, name, fmt=':f'):
        self.name = name  # 评估量名称
        self.fmt = fmt  # 评估量打印格式
        self.val = 0  # 评估量当前值
        self.avg = 0  # 评估量均值
        self.sum = 0  # 历史评估量的和
        self.count = 0  # 历史评估量的数量

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f'{{name}} {{val{self.fmt}}} ({{avg{self.fmt}}})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """ 评估量的进度条打印 """
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = f'{{:{str(num_digits)}d}}'
        return f'[{fmt}/{fmt.format(num_batches)}]'
