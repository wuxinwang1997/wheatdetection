# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

def collate_batch(batch):
    return tuple(zip(*batch))