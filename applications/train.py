# -*- coding: utf-8 -*-
""" 模型训练脚本 """
import time
import shutil
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.optimizer import Optimizer
from apex import amp

from utils import AverageMeter, ProgressMeter
from .test import test


def train(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module,
          criterion: nn.Module, optimizer: Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler, args):
    """ 训练模型
    :param train_loader: 训练集
    :param val_loader: 验证集
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param args: 训练超参
    """
    writer = SummaryWriter(args.logdir)

    best_val_acc1 = 0
    learning_rate = 0
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        learning_rate = scheduler.get_last_lr()
        if isinstance(learning_rate, list):
            learning_rate = learning_rate[0]
        # 训练一个epoch，并在验证集上评估
        train_loss, train_acc1 = train_epoch(train_loader, model, criterion,
                                             optimizer, epoch, args)
        val_acc1, val_loss, _ = test(val_loader, model, criterion, args)
        scheduler.step()
        # 保存当前及最好的acc@1的checkpoint
        is_best = val_acc1 > best_val_acc1
        best_val_acc1 = max(val_acc1, best_val_acc1)
        save_checkpoint({
            # 'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.module.state_dict(),
            # 'best_acc1': best_val_acc1,
            # 'optimizer': optimizer.state_dict(),
        }, is_best, args)
        writer.add_scalar('learning rate', learning_rate, epoch)
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc1, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc1, epoch)
        writer.flush()
    writer.close()
    logging.info(f'Training Over with lr={learning_rate}~~')


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """ 训练模型一个epoch的数据
    :param train_loader: 训练集
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param epoch: 当前迭代次数
    :param args: 训练超参
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, prefix=f"Epoch: [{epoch}]")

    # 训练模式
    model.train()
    end_time = time.time()
    for i, (images, scores, _) in enumerate(train_loader):
        # 更新数据加载时间度量
        data_time.update(time.time() - end_time)
        if args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)
            scores = scores.cuda(args.gpu, non_blocking=True)
        # 网络推理
        outputs = model(images)
        loss = criterion(outputs, scores)
        # 计算梯度、更新
        optimizer.zero_grad()
        if args.cuda:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        # 更新度量
        acc1, _, _ = criterion.accuracy(outputs, scores)
        batch_size = images.size(0)/2 if args.criterion == 'rank' else images.size(0)
        losses.update(loss.detach().cpu().item(), batch_size)
        top1.update(acc1.item(), batch_size)
        # 更新一个batch训练时间度量
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.print_freq == 0:
            progress.print(i)
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoints/checkpoint_{}.pth'):
    """ 保存模型
    :param state: 模型状态
    :param is_best: 模型是否当前测试集准确率最高
    :param args: 训练超参
    :param filename: 保存的文件名
    """
    filename = filename.format(args.arch)
    if (args.gpus > 1) and (args.gpu != args.gpus - 1):
        # 同一台服务器上多卡训练时，只有最后一张卡保存模型（多卡同时保存到同一位置会混乱）
        return
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'checkpoints/model_best_{args.arch}.pth')
