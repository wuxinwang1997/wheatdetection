# PyTorch Project for global-wheat-detection
This project is based on https://github.com/L1aoXingyu/Deep-Learning-Project-Template

And I remove the high-leval api ignite which makes the project not easy to change training loop.
# Requirements
- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform) 

# Table Of Contents
-  [In Details](#in-details)
-  [How to run this project](#running)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)


# In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── train_mnist_softmax.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
│    └── build.py  		   - here's the file to make dataloader.
│    └── collate_batch.py   - here's the file that is responsible for merges a list of samples to form a mini-batch.
│
│
├──  engine
│   └── average.py   -this file contains the average method.
│   └── fitter.py     - this file contains the train loops and the validation loop contains valid loss output.
|   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of my project.
│   └── _utils.py        -copy of torchvision.models.detection._utils
|   └── backbone_utils.py     -this file is based on torchvision.models.detection.backbone_utils and add resnest101e to it.
|   └── faster_rcnn.py     -copy of torchvision.models.detection.backbone_utils
│   └── generalized_rcnn.py     -this file is based on torchvision.models.detection.generalized_rcnn and add valid loss returning to it.
|   └── image_list.py     -copy of torchvision.models.detection.image_list
|   └── keypoint_rcnn.py     -copy of torchvision.models.detection.keypoint_rcnn
|   └── label_smooth_crossentropy.py     -this file contains the label smooth crossentropy loss
|   └── mask_rcnn.py     -copy of torchvision.models.detection.mask_rcnn
|   └── roi_heads.py     -this file is based on torchvision.models.detection.roi_heads and add valid loss returning to it.
|   └── rpn.py     -this file is based on torchvision.models.detection.rpn and add valid loss returning to it.
|   └── transform.py     -copy of torchvision.models.detection.transform
|
|
├── modeling            - this folder contains any model of my project.
│   └── wheat_detector.py   -this file defines the wheat_detector model based on fasterrcnn
│
│
├── solver             - this folder contains optimizer of my project.
│   └── build.py            -this file contains optimizer function
│   └── lr_scheduler.py       -this file contains lr_scheduler function
│   
│ 
├──  tools                - here's the train/test model of my project.
│    └── train_net.py  - here's the pipeline of train model that is responsible for the whole pipeline.
|    └── test_net.py  - here's the pipeline  of train model that is responsible for the whole pipeline.
│ 
│ 
└── utils
│    ├── logger.py
│    └── any_other_utils_i_need
│ 
│ 
└── tests					- this foler contains unit test of my project.
     ├── test_data_sampler.py
```

# How to run this project
train
```buildoutcfg
python ./tools/train_net.py OUTPUT_DIR '("/content/experiments/")'
```
infer
```buildoutcfg
python ./tools/test_net.py MODEL.PRETRAINED '(False)' DATASETS.ROOT_DIR '("/kaggle/input/global-wheat-detection")' TEST.WEIGHT '("/kaggle/input/resnestwheat/giou-fold0.bin")' OUTPUT_DIR '("/kaggle/working/")'
```

# Future Work

# Contributing
Any kind of enhancement or contribution is welcomed.

# Acknowledgments



