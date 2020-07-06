# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import sys
<<<<<<< HEAD
sys.path.insert(0, "./external/yacs")
=======
sys.path.insert(0, "/content/drive/My Drive/Global_Wheat_Detection/code/wheatdetection/external/yacs")
>>>>>>> d519b2ac861c4c457525c1afe654f42bc2e77079
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.DEBUG = False
_C.SEED = 42
_C.VERBOSE = True

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.PRETRAINED = True

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# RandomSizedCrop paramters
_C.INPUT.RSC_MIN_MAX_HEIGHT = (800, 800)
_C.INPUT.RSC_HEIGHT = 1024
_C.INPUT.RSC_WIDTH = 1024
_C.INPUT.RSC_PROB = 0.5
# HueSaturationValue paramters
_C.INPUT.HSV_H = 0.2
_C.INPUT.HSV_S = 0.2
_C.INPUT.HSV_V = 0.2
_C.INPUT.HSV_PROB = 0.9
# RandomBrightnessContrast paramters
_C.INPUT.BC_B = 0.2
_C.INPUT.BC_C = 0.2
_C.INPUT.BC_PROB = 0.9
# Color paramters
_C.INPUT.COLOR_PROB = 0.9
# Random probability for ToGray
_C.INPUT.TOFGRAY_PROB = 0.01
# Random probability for HorizontalFlip
_C.INPUT.HFLIP_PROB = 0.5
# Random probability for VerticalFlip
_C.INPUT.VFLIP_PROB = 0.5
# Coutout paramters
_C.INPUT.COTOUT_NUM_HOLES = 8
_C.INPUT.COTOUT_MAX_H_SIZE = 64
_C.INPUT.COTOUT_MAX_W_SIZE = 64
_C.INPUT.COTOUT_FILL_VALUE = 0
_C.INPUT.COTOUT_PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# Root dir of dataset
_C.DATASETS.ROOT_DIR = "/content/global-wheat-detection"
# Fold to validate
_C.DATASETS.VALID_FOLD = 0
# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# # List of the dataset names for testing, as present in paths_catalog.py
# _C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 2

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.SCHEDULER_NAME = "CosineAnnealingWarmRestarts"
_C.SOLVER.COS_CPOCH = 2
_C.SOLVER.T_MUL = 2

_C.SOLVER.MAX_EPOCHS = 40

_C.SOLVER.BASE_LR = 0.005
_C.SOLVER.BIAS_LR_FACTOR = 1

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.WEIGHT_DECAY_BN = 0

_C.SOLVER.WARMUP_EPOCHS = 10

_C.SOLVER.EARLY_STOP_PATIENCE = 20

_C.SOLVER.TRAIN_CHECKPOINT = False
_C.SOLVER.CLEAR_OUTPUT = 10

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 4

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 4
_C.TEST.WEIGHT = "/content/output/best-checkpoint.bin"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "/content/drive/My Drive/Global_Wheat_Detection/experiments/baseline"
