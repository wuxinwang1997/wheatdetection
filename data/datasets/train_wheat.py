# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class train_wheat(Dataset):

    def __init__(self, root, marking, image_ids, transforms=None, test=False):
        super().__init__()
        self.root = root
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        p_ratio = random.random()
        if self.test or p_ratio > 0.6:
            image, boxes = self.load_image_and_boxes(index)
        else:
            if p_ratio < 0.2:
                image, boxes = self.load_mosaic_image_and_boxes(index)
            elif p_ratio < 0.4:
                image, boxes = self.load_image_and_bboxes_with_cutmix(index)
            else:
                image, boxes = self.load_mixup_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    # target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.root}/train/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_mosaic_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of mosaic author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

    def load_image_and_bboxes_with_cutmix(self, index):
        image, bboxes = self.load_image_and_boxes(index)
        image_to_be_mixed, bboxes_to_be_mixed = self.load_image_and_boxes(
            random.randint(0, self.image_ids.shape[0] - 1))

        image_size = image.shape[0]
        cutoff_x1, cutoff_y1 = [int(random.uniform(image_size * 0.0, image_size * 0.49)) for _ in range(2)]
        cutoff_x2, cutoff_y2 = [int(random.uniform(image_size * 0.5, image_size * 1.0)) for _ in range(2)]

        image_cutmix = image.copy()
        image_cutmix[cutoff_y1:cutoff_y2, cutoff_x1:cutoff_x2] = image_to_be_mixed[cutoff_y1:cutoff_y2,
                                                                 cutoff_x1:cutoff_x2]

        # Begin preparing bboxes_cutmix.
        # Case 1. Bounding boxes not intersect with cut off patch.
        bboxes_not_intersect = bboxes[np.concatenate((np.where(bboxes[:, 0] > cutoff_x2),
                                                      np.where(bboxes[:, 2] < cutoff_x1),
                                                      np.where(bboxes[:, 1] > cutoff_y2),
                                                      np.where(bboxes[:, 3] < cutoff_y1)), axis=None)]

        # Case 2. Bounding boxes intersect with cut off patch.
        bboxes_intersect = bboxes.copy()

        top_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                 (bboxes[:, 2] > cutoff_x1) &
                                 (bboxes[:, 1] < cutoff_y2) &
                                 (bboxes[:, 3] > cutoff_y2))
        right_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                   (bboxes[:, 2] > cutoff_x2) &
                                   (bboxes[:, 1] < cutoff_y2) &
                                   (bboxes[:, 3] > cutoff_y1))
        bottom_intersect = np.where((bboxes[:, 0] < cutoff_x2) &
                                    (bboxes[:, 2] > cutoff_x1) &
                                    (bboxes[:, 1] < cutoff_y1) &
                                    (bboxes[:, 3] > cutoff_y1))
        left_intersect = np.where((bboxes[:, 0] < cutoff_x1) &
                                  (bboxes[:, 2] > cutoff_x1) &
                                  (bboxes[:, 1] < cutoff_y2) &
                                  (bboxes[:, 3] > cutoff_y1))

        # Remove redundant indices. e.g. a bbox which intersects in both right and top.
        right_intersect = np.setdiff1d(right_intersect, top_intersect)
        right_intersect = np.setdiff1d(right_intersect, bottom_intersect)
        right_intersect = np.setdiff1d(right_intersect, left_intersect)
        bottom_intersect = np.setdiff1d(bottom_intersect, top_intersect)
        bottom_intersect = np.setdiff1d(bottom_intersect, left_intersect)
        left_intersect = np.setdiff1d(left_intersect, top_intersect)

        bboxes_intersect[:, 1][top_intersect] = cutoff_y2
        bboxes_intersect[:, 0][right_intersect] = cutoff_x2
        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1
        bboxes_intersect[:, 2][left_intersect] = cutoff_x1

        bboxes_intersect[:, 1][top_intersect] = cutoff_y2
        bboxes_intersect[:, 0][right_intersect] = cutoff_x2
        bboxes_intersect[:, 3][bottom_intersect] = cutoff_y1
        bboxes_intersect[:, 2][left_intersect] = cutoff_x1

        bboxes_intersect = bboxes_intersect[np.concatenate((top_intersect,
                                                            right_intersect,
                                                            bottom_intersect,
                                                            left_intersect), axis=None)]

        # Case 3. Bounding boxes inside cut off patch.
        bboxes_to_be_mixed[:, [0, 2]] = bboxes_to_be_mixed[:, [0, 2]].clip(min=cutoff_x1, max=cutoff_x2)
        bboxes_to_be_mixed[:, [1, 3]] = bboxes_to_be_mixed[:, [1, 3]].clip(min=cutoff_y1, max=cutoff_y2)

        # Integrate all those three cases.
        bboxes_cutmix = np.vstack((bboxes_not_intersect, bboxes_intersect, bboxes_to_be_mixed)).astype(int)
        bboxes_cutmix = bboxes_cutmix[np.where((bboxes_cutmix[:, 2] - bboxes_cutmix[:, 0]) \
                                               * (bboxes_cutmix[:, 3] - bboxes_cutmix[:, 1]) > 500)]
        # End preparing bboxes_cutmix.

        return image_cutmix, bboxes_cutmix

    def load_mixup_image_and_boxes(self, index):
        image, boxes = self.load_image_and_boxes(index)
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        return (image + r_image) / 2, np.vstack((boxes, r_boxes)).astype(np.int32)