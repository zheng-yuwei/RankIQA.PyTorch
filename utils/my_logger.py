# -*- coding: utf-8 -*-
""" 日志对应 """
import logging
from logging.handlers import RotatingFileHandler


def generate_logger(filename, **log_params):
    """
    生成日志记录对象记录日志
    :param filename: 日志文件名称
    :param log_params: 日志参数
    :return:
    """
    level = log_params.setdefault('level', logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s')
    # 定义一个RotatingFileHandler，最多备份3个日志文件，每个日志文件最大10M
    file_handler = RotatingFileHandler(filename, maxBytes=10 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(formatter)
    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console)
