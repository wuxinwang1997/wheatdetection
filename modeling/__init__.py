# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .weat_detector import WheatDetector


def build_model(cfg):
    model = WheatDetector(cfg)
    return model
