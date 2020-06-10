# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

from torch.utils import data
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .datasets.train_wheat import train_wheat
from .datasets.test_wheat import test_wheat
from .transforms import build_transforms
from .transforms import get_test_transform
from .collate_batch import  collate_batch

def split_dataset(cfg):
    marking = pd.read_csv(f'{cfg.DATASETS.ROOT_DIR}/train.csv')

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:, i]
    marking.drop(columns=['bbox'], inplace=True)
    marking['area'] = marking['w'] * marking['h']
    marking = marking[marking['area'] < 154200.0]
    error_bbox = [100648.0, 145360.0, 149744.0, 119790.0, 106743.0]
    marking = marking[~marking['area'].isin(error_bbox)]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    train_ids = df_folds[df_folds['fold'] != cfg.DATASETS.VALID_FOLD].index.values
    valid_ids = df_folds[df_folds['fold'] == cfg.DATASETS.VALID_FOLD].index.values
    if cfg.DEBUG:
        train_ids = train_ids[:40]
        valid_ids = valid_ids[:10]

    return marking, train_ids, valid_ids

def build_dataset(cfg):
    marking, train_ids, valid_ids = split_dataset(cfg)
    train_dataset = train_wheat(
        root = cfg.DATASETS.ROOT_DIR,
        image_ids=train_ids,
        marking=marking,
        transforms=build_transforms(cfg, is_train=True),
        test=False,
    )

    validation_dataset = train_wheat(
        root=cfg.DATASETS.ROOT_DIR,
        image_ids=valid_ids,
        marking=marking,
        transforms=build_transforms(cfg, is_train=False),
        test=True,
    )

    return train_dataset, validation_dataset

def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.SOLVER.IMS_PER_BATCH
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH

    train_dataset, validation_dataset = build_dataset(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    val_loader = data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_batch,
    )

    return train_loader, val_loader

def build_test_dataset(cfg):
    test_df = pd.read_csv(f'{cfg.DATASETS.ROOT_DIR}/sample_submission.csv')
    print(test_df.shape)
    test_dataset = test_wheat(test_df, f'{cfg.DATASETS.ROOT_DIR}/test', get_test_transform())

    return test_dataset

def make_test_data_loader(cfg):
    batch_size = cfg.TEST.IMS_PER_BATCH

    test_dataset = build_test_dataset(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_batch
    )

    return test_loader
