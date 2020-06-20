# -*- coding: utf-8 -*-
""" 主入口 """
import os
import random
import datetime
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import models as my_models
from config import parser
import utils
import dataloader
import applications
import criterions


def main():
    """
    3种运行方式：
    1. 单CPU运行模式；
    2. 单GPU运行模式；
    3. 分布式运行模式：多机多卡 或 单机多卡。
    分布式优势：1.支持同步BN； 2.DDP每个训练有独立进程管理，训练速度更快，显存均衡；
    """
    args = parser.parse_args()
    # 根据训练机器和超参，选择运行方式
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available == 0:
        args.gpus = 0
    elif args.gpus > num_gpus_available:
        raise ValueError(f'--gpus(-g {args.gpus}) can not greater than available device({num_gpus_available})')

    # 根据每个节点的GPU数量调整world size
    args.world_size = args.gpus * args.nodes
    if not args.cuda or args.world_size == 0:
        # 1. cpu运行模式
        args.cuda = False
        args.gpus = 0
        args.distributed = False
    elif args.world_size == 1:
        # 2. 单GPU运行模式
        args.distributed = False
    elif args.world_size > 1:
        # 3. 分布式运行模式
        args.distributed = True
    else:
        raise ValueError(f'Check config parameters --nodes/-n={args.nodes} and --gpus/-g={args.gpus}!')

    if args.distributed and args.gpus > 1:
        # use torch.multiprocessing.spawn to launch distributed processes
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(gpu, args):
    """ 模型训练、测试、转JIT、蒸馏文件制作
    :param gpu: 运行的gpu id
    :param args: 运行超参
    """
    args.gpu = gpu
    utils.generate_logger(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{gpu}.log")
    logging.info(f'args: {args}')

    # 可复现性
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if args.cuda:
        logging.info(f"Use GPU: {args.gpu} ~")
        if args.distributed:
            args.rank = args.rank * args.gpus + gpu
            dist.init_process_group(backend='nccl', init_method=args.init_method,
                                    world_size=args.world_size, rank=args.rank)
    else:
        logging.info(f"Use CPU ~")

    # 创建/加载模型，使用预训练模型时，需要自己先下载好放到 pretrained 文件夹下，以网络名词命名
    logging.info(f"=> creating model '{args.arch}'")
    model = my_models.get_model(args.arch, args.pretrained, num_classes=args.num_classes)

    # 重加载之前训练好的模型
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            acc = model.load_state_dict(checkpoint['state_dict'])
            logging.info(f'missing keys of models: {acc.missing_keys}')
            del checkpoint
        else:
            raise Exception(f"No checkpoint found at '{args.resume}' to be resumed")

    # 模型信息
    image_height, image_width = args.image_size
    logging.info(f'Model {args.arch} input size: ({image_height}, {image_width})')
    utils.summary(size=(image_height, image_width), channel=3, model=model)

    # 模型转换：转为 torch.jit.script
    if args.jit:
        if not args.resume:
            raise Exception('Option --resume must specified!')
        applications.convert_to_jit(model, args=args)
        return

    if args.criterion == 'rank':
        criterion = criterions.RankingLoss(args=args)  # 对比排序损失
    elif args.criterion == 'emd':
        criterion = criterions.EMDLoss()  # 推土机距离损失
    elif args.criterion == 'regress':
        criterion = criterions.RegressionLoss()  # MSE回归损失
    else:
        raise NotImplementedError(f'Not loss function {args.criterion}，only (rank, emd, regress)!')

    if args.cuda:
        if args.distributed and args.sync_bn:
            model = apex.parallel.convert_syncbn_model(model)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)

    # 优化器：Adam > SGD > SWA(SGD > Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 可尝试优化器
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.lr, momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # from optim.torchtools.optim import RangerLars, Ralamb, Novograd, LookaheadAdam, Ranger, RAdam, AdamW
    # optimizer = RangerLars(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Ralamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Novograd(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = LookaheadAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = Ranger(model_params, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 随机均值平均优化器
    # from optim.swa import SWA
    # optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

    # 混合精度训练
    if args.cuda:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        model = DDP(model)
    else:
        model = torch.nn.DataParallel(model)

    if args.train:
        train_loader = dataloader.load(args, 'train')
        val_loader = dataloader.load(args, 'val')
        scheduler = LambdaLR(optimizer,
                             lambda epoch: adjust_learning_rate(epoch, args=args))
        applications.train(train_loader, val_loader, model, criterion, optimizer, scheduler, args)
        args.evaluate = True

    if args.evaluate:
        torch.set_flush_denormal(True)
        test_loader = dataloader.load(args, name='test')
        acc, loss, test_results = applications.test(test_loader, model, criterion, args)
        logging.info(f'Evaluation: * Acc@1 {acc:.3f} and loss {loss:.3f}.')
        logging.info(f'Evaluation results:')
        for result in test_results:
            logging.info(' '.join([str(r) for r in result]))
        logging.info('Evaluation Over~')


def adjust_learning_rate(epoch, args):
    """ 根据warmup设置、迭代代数、设置的学习率，调整每一代的学习率
    :param epoch: 当前epoch数
    :param args: 使用warmup代数、学习率
    """
    # lr_rates = [0.1, 1., 10., 100., 1e-10]
    # epochs = [2, 4, 6, 8, 10]
    epoch_step = (args.epochs - args.warmup) / 4.0
    lr_rates = np.array([0.1, 1., 0.1, 0.01, 0.001])
    epochs = np.array([args.warmup,
                       args.warmup + int(1.5 * epoch_step),
                       args.warmup + int(2.5 * epoch_step),
                       args.warmup + int(3.5 * epoch_step),
                       args.epochs])
    for i, e in enumerate(epochs):
        if e > epoch:
            return lr_rates[i]
        elif e == epoch:
            next_rate = lr_rates[i]
            if len(lr_rates) > i + 1:
                next_rate = lr_rates[i + 1]
            logging.info(f'===== lr decay rate: {lr_rates[i]} -> {next_rate} =====')

    return lr_rates[-1]


if __name__ == '__main__':
    main()
