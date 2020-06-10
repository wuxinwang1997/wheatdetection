# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import argparse
import os
import sys
from os import mkdir

import torch

from os import mkdir
sys.path.append('/content/drive/My Drive/global-wheat-detection/code/wheatdetection/')
from config import cfg
from data import make_test_data_loader
from modeling import build_model
from utils.logger import setup_logger
from engine.tester import Tester

def test(cfg):
    model = build_model(cfg)
    device = cfg.MODEL.DEVICE
    checkpoint = torch.load(cfg.TEST.WEIGHT)
    best_score_threshold = checkpoint['best_score_threshold']
    best_final_score = checkpoint['best_final_score']
    print('-' * 30)
    print(f'[Best Score Threshold]: {best_score_threshold}')
    print(f'[OOF Score]: {best_final_score:.4f}')
    print('-' * 30)
    test_loader = make_test_data_loader(cfg)

    tester = Tester(model=model, device=device, cfg=cfg, test_loader=test_loader)
    tester.test()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Inference")
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
    test(cfg)

if __name__ == '__main__':
    main()
