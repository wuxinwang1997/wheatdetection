# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import product
import sys
sys.path.insert(0, "./external/wbf")
import ensemble_boxes
warnings.filterwarnings("ignore")


class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 1024

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes):
        raise NotImplementedError


class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return boxes


class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [3, 1]]
        res_boxes[:, [1, 3]] = boxes[:, [0, 2]]
        return res_boxes


class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """

    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)

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

    def process_det(self, index, outputs, score_threshold=0.5):
        boxes = outputs[index]['boxes'].data.cpu().numpy()
        scores = outputs[index]['scores'].data.cpu().numpy()
        boxes = (boxes).clip(min=0, max=1023).astype(int)
        indexes = np.where(scores > score_threshold)
        boxes = boxes[indexes]
        scores = scores[indexes]
        return boxes, scores

    def make_tta_predictions(self, tta_transforms, images, score_threshold=0.5):
        with torch.no_grad():
            images = torch.stack(images).float().cuda()
            predictions = []
            for tta_transform in tta_transforms:
                result = []
                outputs = self.model(tta_transform.batch_augment(images.clone()))

                for i, image in enumerate(images):
                    boxes = outputs[i]['boxes'].data.cpu().numpy()
                    scores = outputs[i]['scores'].data.cpu().numpy()
                    indexes = np.where(scores > score_threshold)[0]
                    boxes = boxes[indexes]
                    boxes = tta_transform.deaugment_boxes(boxes.copy())
                    result.append({
                        'boxes': boxes,
                        'scores': scores[indexes],
                    })
                predictions.append(result)
        return predictions

    def run_wbf(self, predictions, image_index, image_size=1024, iou_thr=0.5, skip_box_thr=0.43, weights=None):
        boxes = [(prediction[image_index]['boxes'] / (image_size - 1)).tolist() for prediction in predictions]
        scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
        labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in
                  predictions]
        boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels,
                                                                                        weights=None, iou_thr=iou_thr,
                                                                                        skip_box_thr=skip_box_thr)
        boxes = boxes * (image_size - 1)
        return boxes, scores, labels

    def format_prediction_string(self, boxes, scores):
        pred_strings = []
        for j in zip(scores, boxes):
            pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
        return " ".join(pred_strings)

    def infer(self):
        self.model.eval()
        torch.cuda.empty_cache()

        tta_transforms = []
        for tta_combination in product([TTAHorizontalFlip(), None],
                                       [TTAVerticalFlip(), None],
                                       [TTARotate90(), None]):
            tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
        test_loader = tqdm(self.test_loader, total=len(self.test_loader), desc="Testing")
        results = []

        for images, image_ids in test_loader:
            predictions = self.make_tta_predictions(tta_transforms, images)
            for i, image in enumerate(images):
                boxes, scores, labels = self.run_wbf(predictions, image_index=i)
                boxes = boxes.round().astype(np.int32).clip(min=0, max=1023)
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

    def save_predictions(self, results):
        test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
        test_df.to_csv(f'{self.config.OUTPUT_DIR}/submission.csv', index=False)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def log(self, message):
        if self.config.VERBOSE:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')