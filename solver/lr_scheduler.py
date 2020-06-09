# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import torch

def make_scheduler(cfg, optimizer, train_loader):
    number_of_iteration_per_epoch = len(train_loader)
    learning_rate_step_size = cfg.SOLVER.COS_CPOCH * number_of_iteration_per_epoch
    scheduler = getattr(torch.optim.lr_scheduler, cfg.SOLVER.SCHEDULER_NAME)(optimizer, T_0=learning_rate_step_size, T_mult=cfg.SOLVER.T_MUL)
    return scheduler