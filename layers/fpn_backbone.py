# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import sys
sys.path.insert(0, "/content/drive/My Drive/global-wheat-detection/code/wheatdetection/external/pytorch-image-models")
from timm.models.resnest import resnest101e
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops

def fpn_backbone(pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=3):
    # backbone = resnet.__dict__['resnet18'](pretrained=pretrained,norm_layer=norm_layer)
    backbone = resnest101e(pretrained=pretrained)
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)