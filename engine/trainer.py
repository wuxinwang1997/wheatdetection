# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import random
import os
import numpy as np
import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from evaluate import inference

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_supervised_trainer(model, optimizer, device=None):
    """
    Factory function for creating a trainer for supervised models
    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        images, targets, image_ids = batch
        images = torch.stack(images)
        images = images.to(device).float()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for i in range(len(targets)):
            targets[i]['boxes'] = targets[i]['boxes'].float()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

def create_supervised_evaluator(model, all_predictions, device=None):
    """
    Factory function for creating an evaluator for supervised models
    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            images, targets, image_ids = batch
            images = list(image.cuda() for image in images)
            outputs = model(images)
            inference(all_predictions, images, outputs, targets, image_ids)
            return all_predictions

    engine = Engine(_inference)

    return engine

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    seed_everything(cfg.SEED)
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    all_predictions = []
    trainer = create_supervised_trainer(model, optimizer, device=device)
    evaluator = create_supervised_evaluator(model, all_predictions, device=device)
    checkpointer = ModelCheckpoint(output_dir, 'mnist', checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['ce_loss']
        logger.info("Training Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['ce_loss']
            logger.info("Validation Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                        .format(engine.state.epoch, avg_accuracy, avg_loss)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
