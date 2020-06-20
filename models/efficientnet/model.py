# -*- coding: utf-8 -*-
import typing
from functools import partial

from torch import nn

from .config import BlockArgs, GlobalParams, EfficientNetConfig
from .utils import round_filters, round_repeats, load_pretrained_weights
from .components import Swish, Conv2dStaticSamePadding, Identity, MBConvBlock


class EfficientNet(nn.Module):
    """ EfficientNet模型 """

    def __init__(self, blocks_args: typing.List[BlockArgs] = None,
                 global_params: GlobalParams = None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv2d = partial(Conv2dStaticSamePadding, image_size=global_params.image_size)

        # BN层超参
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # 主干网络
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList()
        block_args = None
        for block_args in self._blocks_args:

            # 根据网络超参更新模块的输入/输出卷积核数
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # 第一个block需要处理好stride和卷积核尺寸
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # 最后线性全连接层
        self._swish = Swish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        # sample-free
        # pos_bias = np.array([47775, 377470, 22176, 21935, 21584], dtype=np.float32)  # 每一类的正类数量
        # neg_bias = np.array([47775, 100000, 22176, 21935, 21584], dtype=np.float32)  # 每一类的负类数量
        # bias = -np.log(neg_bias / pos_bias)
        # self._fc.bias = torch.nn.Parameter(torch.from_numpy(bias))

    def extract_features(self, inputs):
        """ 得到最后一层卷积层的输出 """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ 调用extract_features函数抽取最后一层特征，接上全连接层，得到logits输出 """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, **override_params):
        """ 根据EfficientNet系列相应的名字构建模型
        :param model_name: EfficientNet系列相应的名字
        :param override_params: 模型的一些自定义超参
        """
        EfficientNetConfig.check_model_name_is_valid(model_name)
        blocks_args, global_params = EfficientNetConfig.get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3, adv_prop=False):
        """ 加载预训练模型
        :param model_name: 模型名称
        :param num_classes: 模型类别数
        :param in_channels: 输入图像通道数
        :param adv_prop: 使用adv_prop预训练模型
        """
        model = cls.from_name(model_name, num_classes=num_classes)
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), adv_prop=adv_prop)
        if in_channels != 3:
            Conv2d = partial(Conv2dStaticSamePadding, image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """ 获取图像输入尺寸
        :param model_name: 模型名称
        """
        EfficientNetConfig.check_model_name_is_valid(model_name)
        _, _, res, _ = EfficientNetConfig.get_search_params(model_name)
        return res
