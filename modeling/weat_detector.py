# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn
from torchvision.models.detection import FasterRCNN
from layers.fpn_backbone import fpn_backbone

class WheatDetector(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(WheatDetector, self).__init__()
        self.backbone = fpn_backbone(pretrained=True)
        self.base = FasterRCNN(self.backbone, num_classes = cfg.MODEL.NUM_CLASSES, **kwargs)
        # self.base.roi_heads.fastrcnn_loss = fastrcnn_loss

    def forward(self, images, targets=None):
        return self.base(images, targets)
