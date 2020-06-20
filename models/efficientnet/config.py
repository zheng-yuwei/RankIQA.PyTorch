# -*- coding: utf-8 -*-
""" EfficientNet系列配置 """
import re
import typing
import collections


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon',
    'dropout_rate', 'drop_connect_rate',
    'width_coefficient', 'depth_coefficient',
    'image_size', 'num_classes',
    'depth_divisor', 'min_depth'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class EfficientNetConfig:

    # 不同大小的EfficientNet的网络超参
    SEARCH_PARAMS = {
        # Coefficients:   width, depth, res, dropout
        'efficientnet-b0': (1.0, 1.0, [224, 224], 0.2),
        'efficientnet-b1': (1.0, 1.1, [240, 240], 0.2),
        'efficientnet-b2': (1.1, 1.2, [260, 260], 0.3),
        'efficientnet-b3': (1.2, 1.4, [300, 300], 0.3),
        'efficientnet-b4': (1.4, 1.8, [380, 380], 0.4),
        'efficientnet-b5': (1.6, 2.2, [456, 456], 0.4),
        'efficientnet-b6': (1.8, 2.6, [528, 528], 0.5),
        'efficientnet-b7': (2.0, 3.1, [600, 600], 0.5),
        'efficientnet-b8': (2.2, 3.6, [672, 672], 0.5),
        'efficientnet-l2': (4.3, 5.3, [800, 800], 0.5),
    }

    # EfficientNet的7个block的结构超参
    BLOCKS_ARGS = [
        # num_repeat, kernel_size, stride, expand_ratio,
        # input_filters, output_filters, se_ratio, id_skip（默认为True，noskip为False）
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]

    @classmethod
    def check_model_name_is_valid(cls, model_name):
        """ 检查EfficientNet模型名是否有效 """
        if model_name not in cls.SEARCH_PARAMS:
            raise ValueError(f'model_name should be one of: {cls.SEARCH_PARAMS.keys()}.')

    @classmethod
    def get_model_params(cls, model_name, override_params):
        """ 根据模型名称，获取该模型的基础结构blocks_args、模型超参global_params """
        if model_name.startswith('efficientnet'):
            w, d, s, p = cls.get_search_params(model_name)
            # note: all models have drop connect rate = 0.2
            blocks_args, global_params = cls._get_model_params(
                width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
        else:
            raise NotImplementedError(f'model name is not pre-defined: {model_name}')

        if override_params:
            global_params = global_params._replace(**override_params)
        return blocks_args, global_params

    @classmethod
    def get_search_params(cls, model_name: str) -> \
            typing.Tuple[float, float, typing.List[int], float]:
        """ 获取对应EfficientNet结构搜索出来的超参
        :param model_name:
        :return EfficientNet搜索超参
        """
        return cls.SEARCH_PARAMS[model_name]

    @classmethod
    def _get_model_params(
            cls, width_coefficient: float = None, depth_coefficient: float = None,
            dropout_rate: float = 0.2, drop_connect_rate: float = 0.2,
            image_size: typing.List[int] = None, num_classes: int = 1000
    ) -> (typing.List[BlockArgs], GlobalParams):
        """ 根据模型超参，构造 每个block的超参的列表、整体模型的超参
        :param width_coefficient: block宽度系数
        :param depth_coefficient: block深度系数
        :param dropout_rate: 整体网络fc全连接层的dropout系数
        :param drop_connect_rate: 卷积层drop connection系数
        :param image_size: 输入图像分辨率，[H, W]
        :param num_classes: 网络输出的分支数
        :returns
            blocks_args: 每个block的参数

        """
        blocks_args = BlockDecoder.decode(cls.BLOCKS_ARGS)

        global_params = GlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=dropout_rate,
            drop_connect_rate=drop_connect_rate,
            num_classes=num_classes,
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            depth_divisor=8,
            min_depth=None,
            image_size=image_size,
        )

        return blocks_args, global_params


class BlockDecoder:
    """ 解析efficient net基础block的参数 """

    @staticmethod
    def _decode_block_string(block_string: str) -> BlockArgs:
        """ 解析block超参的字符串为BlockArgs变量
        :param block_string: block超参的字符串
        :return
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block: BlockArgs) -> str:
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list: typing.List[str]) -> typing.List[BlockArgs]:
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args: typing.List[BlockArgs]) -> typing.List[str]:
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings
