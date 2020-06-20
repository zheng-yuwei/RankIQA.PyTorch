# -*- coding: utf-8 -*-
""" 损失函数
RankingLoss: 排序损失
EMDLoss: 推土距离损失
RegressionLoss: MSE回归损失
"""
from .ranking_loss import RankingLoss
from .regress_loss import RegressionLoss
from .emd_loss import EMDLoss
