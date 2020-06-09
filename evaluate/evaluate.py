# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
from tqdm import tqdm
import numpy as np
from .calculate_score import calculate_final_score

def evaluate(all_predictions):
    best_final_score, best_score_threshold = 0, 0
    for score_threshold in tqdm(np.arange(0, 1, 0.01), total=np.arange(0, 1, 0.01).shape[0], desc="OOF"):
        final_score = calculate_final_score(all_predictions, score_threshold)
        if final_score > best_final_score:
            best_final_score = final_score
            best_score_threshold = score_threshold

    for i in range(len(all_predictions)):
        gt_boxes = all_predictions[i]['gt_boxes'].copy()
        pred_boxes = all_predictions[i]['pred_boxes'].copy()
        scores = all_predictions[i]['scores'].copy()
        indexes = np.where(scores>best_score_threshold)
        pred_boxes = pred_boxes[indexes]
        all_predictions[i]['final_missed_boxes_nums'] = len(gt_boxes)-len(pred_boxes)

    return best_score_threshold, best_final_score