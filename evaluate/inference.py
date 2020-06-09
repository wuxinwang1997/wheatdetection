# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import numpy as np

def inference(all_predictions, batch_images, outputs, targets, image_ids):
    for i, image in enumerate(batch_images):
        boxes = outputs[i]['boxes'].data.cpu().numpy().astype(np.int32)
        scores = outputs[i]['scores'].data.cpu().numpy()
        # boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        # boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

        all_prediction = {
            'pred_boxes': boxes,
            'scores': scores,
            'gt_boxes': targets[i]['boxes'].cpu().numpy().astype(int),
            'image_id': image_ids[i],
        }

        all_predictions.append(all_prediction)