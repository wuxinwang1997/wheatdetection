# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from .wheat_detector import WheatDetector


def build_model(cfg):
    model = WheatDetector(cfg)
    return model
