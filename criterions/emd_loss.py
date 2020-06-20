# -*- coding: utf-8 -*-
""" EMDLoss """
import torch
from torch import nn


class EMDLoss(nn.Module):
    """ 推土距离损失 """

    def __init__(self):
        """ 推土距离损失 """
        super(EMDLoss, self).__init__()

    def forward(self, outputs, targets):
        """ 网络输出和DMOS分数的normalized=2的推土距离损失
        :param outputs: 图像的网络输出
        :param targets: DMOS分数label
        :return EMD损失
        """
        cdf_outputs = torch.cumsum(outputs.softmax(dim=1), dim=1)
        cdf_targets = torch.cumsum(targets, dim=1)
        distances = torch.sqrt(torch.mean(torch.square(cdf_outputs - cdf_targets), dim=1))
        return torch.mean(distances)

    @staticmethod
    def accuracy(outputs: torch.FloatTensor, targets: torch.FloatTensor) -> \
            (torch.float32, torch.IntTensor, torch.FloatTensor):
        """ 网络输出和DMOS分数的准确率等指标计算
        :param outputs: 图像的网络输出
        :param targets: DMOS分数label
        :return 按概率求均值后的分值误差 torch.float32，
                网络预测的最大概率类别(batch_size)，
                预测的概率分布(batch_size, num_classes)
        """
        with torch.no_grad():
            probs = outputs.softmax(dim=1)
            _, pred = outputs.max(dim=1)
            acc = torch.sum((probs - targets) * torch.arange(1, targets.size(1) + 1), dim=1).abs().mean()
            return acc, pred, probs


if __name__ == '__main__':
    import numpy as np
    emd_loss = EMDLoss()
    y_pred = torch.from_numpy(np.array([[0.5, 0.25, 0.25],
                                        [0.5, 0.0, 0.5],
                                        [0.25, 0.25, 0.5],
                                        [0.25, 0.5, 0.25],
                                        [0.5, 0.5, 0.0],
                                        [0., 0.5, 0.5]]))  # torch.rand((5, 10))
    y_label = torch.from_numpy(np.array([[0.5, 0.0, 0.5],
                                         [0.5, 0.25, 0.25],
                                         [0.25, 0.5, 0.25],
                                         [0.25, 0.25, 0.5],
                                         [0., 0.5, 0.5],
                                         [0.5, 0.5, 0.0]]))  # torch.rand((5, 10)).softmax(dim=1)
    my_loss = emd_loss(y_pred, y_label)
    print(f'my_loss: {my_loss}')
    my_acc, my_pred, my_probs = emd_loss.accuracy(y_pred, y_label)
    print(f'acc: {my_acc.detach().cpu().item()}, \n'
          f'pred: {my_pred.detach().cpu().numpy()}, \n'
          f'probs: {my_probs.detach().cpu().numpy()}')
