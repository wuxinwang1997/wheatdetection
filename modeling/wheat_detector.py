# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
import torchvision.models.detection._utils as det_utils
from layers.fpn_backbone import fpn_backbone
from layers.label_smooth_crossentropy import CrossEntropyLabelSmooth
from layers.giou import  generalized_iou_loss as giou

class WheatDetector(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(WheatDetector, self).__init__()
        self.backbone = fpn_backbone(pretrained=cfg.MODEL.PRETRAINED)
        self.base = FasterRCNN(self.backbone, num_classes = cfg.MODEL.NUM_CLASSES, rpn_fg_iou_thresh=0.5, **kwargs)
        self.base.roi_heads.fastrcnn_loss = self.fastrcnn_loss

    def fastrcnn_loss(self, class_logits, box_regression, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.
        Arguments:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)
        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        labal_smooth_loss = CrossEntropyLabelSmooth(2)
        classification_loss = labal_smooth_loss(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = giou(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss*0.05

    def forward(self, images, targets=None):
        return self.base(images, targets)
