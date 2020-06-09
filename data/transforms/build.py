# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .transforms import RandomErasing

def get_train_transforms(cfg):
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=cfg.INPUT.RSC_MIN_MAX_HEIGHT, height=cfg.INPUT.RSC_HEIGHT,
                              width=cfg.INPUT.RSC_WIDTH, p=cfg.INPUT.RSC_PROB),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=cfg.INPUT.HSV_H, sat_shift_limit=cfg.INPUT.HSV_S,
                                     val_shift_limit=cfg.INPUT.HSV_V, p=cfg.INPUT.HSV_PROB),
                A.RandomBrightnessContrast(brightness_limit=cfg.INPUT.BC_B,
                                           contrast_limit=cfg.INPUT.BC_C, p=cfg.INPUT.BC_PROB),
            ],p=cfg.INPUT.COLOR_PROB),
            A.ToGray(p=cfg.INPUT.TOFGRAY_PROB),
            A.HorizontalFlip(p=cfg.INPUT.HFLIP_PROB),
            A.VerticalFlip(p=cfg.INPUT.VFLIP_PROB),
            # A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=cfg.INPUT.COTOUT_NUM_HOLES, max_h_size=cfg.INPUT.COTOUT_MAX_H_SIZE,
                     max_w_size=cfg.INPUT.COTOUT_MAX_W_SIZE, fill_value=cfg.INPUT.COTOUT_FILL_VALUE, p=cfg.INPUT.COTOUT_PROB),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms(cfg):
    return A.Compose(
        [
            # A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = get_train_transforms(cfg)
    else:
        transform = get_valid_transforms(cfg)

    return transform
