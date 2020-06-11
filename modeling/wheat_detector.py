# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
from layers import FasterRCNN
from layers.backbone_utils import resnest_fpn_backbone

class WheatDetector(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(WheatDetector, self).__init__()
        self.backbone = resnest_fpn_backbone(pretrained=True)
        self.base = FasterRCNN(self.backbone, num_classes=cfg.MODEL.NUM_CLASSES, **kwargs)

    def forward(self, images, targets=None):
        return self.base(images, targets)
