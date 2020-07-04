# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import argparse
import os
import sys
from os import mkdir
# sys.path.append('/content/drive/My Drive/global-wheat-detection/code/wheatdetection/')
sys.path.append('/kaggle/input/resnest/wheatdetection/wheatdetection/')
from config import cfg
from data import make_data_loader
from engine.fitter import Fitter
from modeling import build_model
from solver import make_optimizer
import random
import torch
import numpy as np
from utils.logger import setup_logger

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(cfg, logger):
    seed_everything(cfg.SEED)
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE
    check = cfg.SOLVER.TRAIN_CHECKPOINT

    train_loader, val_loader = make_data_loader(cfg, is_train=True)

    fitter = Fitter(model=model, device=device, cfg=cfg, train_loader=train_loader, val_loader=val_loader, logger=logger)
    if check:
        fitter.load(f'{cfg.OUTPUT_DIR}/last-checkpoint.bin')
    fitter.fit()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("wheatdetection", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, logger)


if __name__ == '__main__':
    main()
