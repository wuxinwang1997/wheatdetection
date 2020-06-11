# encoding: utf-8
"""
@author:  wuxin.wang
@contact: wuxin.wang@whu.edu.cn
"""

import os
import time
import warnings
from datetime import datetime
import torch
from .average import AverageMeter
from evaluate.inference import inference
from evaluate.evaluate import evaluate
from tqdm import tqdm
import pandas as pd
from solver.build import make_optimizer
from solver.lr_scheduler import make_scheduler
import logging
from google.colab import output
warnings.filterwarnings("ignore")

class Fitter:
    def __init__(self, model, device, cfg, train_loader, val_loader, logger):
        self.config = cfg
        self.epoch = 0
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.base_dir = f'{self.config.OUTPUT_DIR}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.logger = logger
        self.best_valid_loss = 100.0

        self.model = model
        self.device = device
        self.model.to(self.device)

        self.optimizer = make_optimizer(cfg, self.model)

        self.scheduler = make_scheduler(cfg, self.optimizer, self.train_loader)

        self.logger.info(f'Fitter prepared. Device is {self.device}')
        self.all_predictions = []
        self.early_stop_epochs = 0
        self.early_stop_patience = self.config.SOLVER.EARLY_STOP_PATIENCE
        self.do_scheduler = True
        self.logger.info("Start training")

    def fit(self):
        for epoch in range(self.epoch, self.config.SOLVER.MAX_EPOCHS ):
            if epoch < self.config.SOLVER.WARMUP_EPOCHS:
                lr_scale = min(1., float(epoch + 1) / float(self.config.SOLVER.WARMUP_EPOCHS))
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr_scale * self.config.SOLVER.BASE_LR
                self.do_scheduler = False
            else:
                self.do_scheduler = True
            if self.config.VERBOSE:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.logger.info(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch()

            self.logger.info(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            best_score_threshold, best_final_score, valid_loss = self.validation()

            self.logger.info( f'[RESULT]: Val. Epoch: {self.epoch}, valid_loss: {valid_loss.avg:.5f}, Best Score Threshold: {best_score_threshold:.2f}, Best Score: {best_final_score:.5f}, time: {(time.time() - t):.5f}')

            if valid_loss.avg < self.best_valid_loss:
                self.best_valid_loss = valid_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint.bin')
                self.save_model(f'{self.base_dir}/best-model.bin')
                self.save_predictions(f'{self.base_dir}/all_predictions.csv')

            self.early_stop(best_final_score)
            if self.early_stop_epochs > self.config.SOLVER.EARLY_STOP_PATIENCE:
                self.logger.info('Early Stopping!')
                break

            if self.epoch % self.config.SOLVER.CLEAR_OUTPUT == 0:
                output.clear()

            self.epoch += 1

    def validation(self):
        self.model.eval()
        valid_loss = AverageMeter()
        loss_box_reg = AverageMeter()
        loss_classifier = AverageMeter()
        loss_objectness = AverageMeter()
        loss_rpn_box_reg = AverageMeter()
        t = time.time()
        self.all_predictions = []
        torch.cuda.empty_cache()
        valid_loader = tqdm(self.val_loader, total=len(self.val_loader), desc="Validating")
        with torch.no_grad():
            for step, (images, targets, image_ids) in enumerate(valid_loader):
                images = torch.stack(images)
                images = images.to(self.device).float()
                batch_size = images.shape[0]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                for i in range(len(targets)):
                    targets[i]['boxes'] = targets[i]['boxes'].float()
                loss_dict, outputs = self.model(images, targets)

                loss = sum(loss for loss in loss_dict.values())

                box_reg = loss_dict['loss_box_reg']
                classifier = loss_dict['loss_classifier']
                objectness = loss_dict['loss_objectness']
                rpn_box_reg = loss_dict['loss_rpn_box_reg']

                inference(self.all_predictions, images, outputs, targets, image_ids)

                valid_loss.update(loss.item(), batch_size)
                loss_box_reg.update(box_reg.item(), batch_size)
                loss_classifier.update(classifier.item(), batch_size)
                loss_objectness.update(objectness.item(), batch_size)
                loss_rpn_box_reg.update(rpn_box_reg.item(), batch_size)

                valid_loader.set_description(f'Valid Step {step}/{len(self.val_loader)}, ' + \
                                             f'valid_loss: {valid_loss.avg:.5f}, ' + \
                                             f'loss_box_reg: {loss_box_reg.avg:.5f}, ' + \
                                             f'loss_classifier: {loss_classifier.avg:.5f}, ' + \
                                             f'loss_objectness: {loss_objectness.avg:.5f}, ' + \
                                             f'loss_rpn_box_reg: {loss_rpn_box_reg.avg:.5f}, ' + \
                                             f'time: {(time.time() - t):.5f}')
        
        best_score_threshold, best_final_score = evaluate(self.all_predictions)

        return best_score_threshold, best_final_score, valid_loss

    def train_one_epoch(self):
        self.model.train()
        summary_loss = AverageMeter()
        loss_box_reg = AverageMeter()
        loss_classifier = AverageMeter()
        loss_objectness = AverageMeter()
        loss_rpn_box_reg = AverageMeter()
        t = time.time()
        train_loader = tqdm(self.train_loader, total=len(self.train_loader), desc="Training")
        for step, (images, targets, image_ids) in enumerate(train_loader):
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            for i in range(len(targets)):
                targets[i]['boxes'] = targets[i]['boxes'].float()
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            box_reg = loss_dict['loss_box_reg']
            classifier = loss_dict['loss_classifier']
            objectness = loss_dict['loss_objectness']
            rpn_box_reg = loss_dict['loss_rpn_box_reg']

            loss.backward()

            summary_loss.update(loss.item(), batch_size)
            loss_box_reg.update(box_reg.item(), batch_size)
            loss_classifier.update(classifier.item(), batch_size)
            loss_objectness.update(objectness.item(), batch_size)
            loss_rpn_box_reg.update(rpn_box_reg.item(), batch_size)
            self.optimizer.step()

            if self.do_scheduler:
                self.scheduler.step()
            train_loader.set_description(f'Train Step {step}/{len(self.train_loader)}, ' + \
                                         f'Learning rate {self.optimizer.param_groups[0]["lr"]}, ' + \
                                         f'summary_loss: {summary_loss.avg:.5f}, ' + \
                                         f'loss_box_reg: {loss_box_reg.avg:.5f}, ' + \
                                         f'loss_classifier: {loss_classifier.avg:.5f}, ' + \
                                         f'loss_objectness: {loss_objectness.avg:.5f}, ' + \
                                         f'loss_rpn_box_reg: {loss_rpn_box_reg.avg:.5f}, ' + \
                                         f'time: {(time.time() - t):.5f}')

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'epoch': self.epoch,
        }, path)

    def save_model(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_valid_loss': self.best_valid_loss,
        }, path)

    def save_predictions(self, path):
        df = pd.DataFrame(self.all_predictions)
        df.to_csv(path, index=False)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.epoch = checkpoint['epoch'] + 1

    def early_stop(self, valid_loss):
        if valid_loss > self.best_valid_loss:
            self.early_stop_epochs += 1
        else:
            self.early_stop_epochs = 0