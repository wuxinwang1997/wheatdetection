# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import time
import warnings
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import cv2
from solver.build import make_optimizer
from solver.lr_scheduler import make_scheduler
import logging
from google.colab import output
warnings.filterwarnings("ignore")

class Tester:
    def __init__(self, model, device, cfg, test_loader):
        self.config = cfg
        self.test_loader = test_loader

        self.base_dir = f'{self.config.OUTPUT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_score_threshold = 0.5

        self.model = model
        self.model.eval()

        self.device = device
        self.model.to(self.device)

        self.log(f'Tester prepared. Device is {self.device}')

    def test(self):
        results = self.infer()
        self.save_predictions(results)


    def infer(self):
        self.model.eval()
        results = []
        torch.cuda.empty_cache()
        test_loader = tqdm(self.test_loader, total=len(self.test_loader), desc="Testing")
        with torch.no_grad():
            for step, (images, image_ids) in enumerate(test_loader):
                images = list(image.to(self.device) for image in images)
                outputs = self.model(images)
                for i, image in enumerate(images):
                    boxes = outputs[i]['boxes'].data.cpu().numpy()
                    scores = outputs[i]['scores'].data.cpu().numpy()

                    boxes = boxes[scores >= self.best_score_threshold].astype(np.int32)
                    scores = scores[scores >= self.best_score_threshold]
                    image_id = image_ids[i]

                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                    result = {
                        'image_id': image_id,
                        'PredictionString': self.format_prediction_string(boxes, scores)
                    }

                    results.append(result)

        return results

    def format_prediction_string(self, boxes, scores):
        pred_strings = []
        for j in zip(scores, boxes):
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

        return " ".join(pred_strings)

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score_threshold': self.best_score_threshold,
            'best_final_score': self.best_final_score,
            'epoch': self.epoch,
        }, path)

    def save_predictions(self, results):
        test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
        test_df.to_csv('submission.csv', index=False)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_score_threshold = checkpoint['best_score_threshold']
        self.best_final_score = checkpoint['best_final_score']

    def log(self, message):
        if self.config.VERBOSE:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')