#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the CNN start phase
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .checkpoints import load_model, save_model
from .eval import metrics_for_eval
from tensorboardX import SummaryWriter
from source.utils import AVGMetrics, TrainHistory, jaccard_coeff, dice_coeff
import logging
import time


def _config_logger(save_path, file_name):
    logger = logging.getLogger("Train-Logger")
    # Checking if the folder logs doesn't exist. If True, we must create it.
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    logger_filename = os.path.join(save_path, f"{file_name}_{str(time.time()).replace('.','')}.log")
    fhandler = logging.FileHandler(filename=logger_filename, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def _train_epoch (model, optimizer, loss_fn, data_loader, c_epoch, t_epoch, device):

    model.train()

    print ("Training...")
    # Setting tqdm to show some information on the screen
    with tqdm(total=len(data_loader), ascii=True, desc='Epoch {}/{}: '.format(c_epoch, t_epoch), ncols=150) as t:


        # Variables to store the avg metrics
        loss_avg = AVGMetrics()
        jacc_avg = AVGMetrics()
        dice_avg = AVGMetrics()

        # Getting the data from the DataLoader generator
        for data in data_loader:

            imgs_batch, mask_batch = data[0].to(device), data[1].to(device)
            seg_pred = model(imgs_batch)

            # Computing loss function
            loss = loss_fn(seg_pred, mask_batch)
            jacc = jaccard_coeff(torch.sigmoid(seg_pred), mask_batch)
            dice = dice_coeff(torch.sigmoid(seg_pred), mask_batch)

            # Getting the avg metrics
            loss_avg.update(loss.item())
            jacc_avg.update(jacc.item())
            dice_avg.update(dice.item())

            # Zero the parameters gradient
            optimizer.zero_grad()

            # Computing gradients and performing the update step
            loss.backward()
            optimizer.step()

            # Updating tqdm
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), jacc='{:05.3f}'.format(jacc_avg()),
                          dice='{:05.3f}'.format(dice_avg()))
            t.update()

    return {"loss": loss_avg(), "jacc": jacc_avg(), "dice": dice_avg() }


def fit_model (model, train_data_loader, val_data_loader, optimizer=None, loss_fn=None, epochs=10,
               epochs_early_stop=None, save_folder=None, initial_model=None, device=None, schedule_lr=None,
               model_name="MyModel", resume_train=False, history_plot=True, min_metric_early_stop=None, best_metric="loss"):


    logger = _config_logger(save_folder, model_name)
    logger.info("Starting the training phase")

    if epochs_early_stop is not None:
        logger.info('Early stopping is set using the number of epochs without improvement')
    if min_metric_early_stop is not None:
        logger.info('Early stopping is set using the min loss as threshold')
    if epochs_early_stop is None and min_metric_early_stop is None:
        logger.info('No early stopping is set')


    if loss_fn is None:
        logger.info('Loss: `nn.BCELoss()` was set as default')
        loss_fn = nn.BCELoss()
    else:
        logger.info('Loss: {}'.format(loss_fn))

    if optimizer is None:
        logger.info('Optimizer: Adam with lr=0.001 was set as default')
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        logger.info('Optimizer: {}'.format(optimizer))


    # Checking if we have a saved model. If we have, load it, otherwise, let's start the model from scratch
    epoch_resume = 0
    if initial_model is not None:
        logger.info("Loading the saved model in {} folder".format(initial_model))

        if resume_train:
            model, optimizer, loss_fn, epoch_resume = load_model(initial_model, model)
            logger.info("Resuming the training from epoch {} ...".format(epoch_resume))
        else:
            model = load_model(initial_model, model)

    else:
        logger.info("The model {} will be trained from scratch".format(model_name))


    # Setting the device(s)
    # If GPU is available, let's move the model to there. If you have more than one, let's use them!
    m_gpu = 0
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # device = torch.device("cuda:" + str(torch.cuda.current_device()))
            m_gpu = torch.cuda.device_count()
            if m_gpu > 1:
                print ("The training will be carry out using {} GPUs:".format(m_gpu))
                for g in range(m_gpu):
                    print (torch.cuda.get_device_name(g))
                    logger.info(torch.cuda.get_device_name(g))

                model = nn.DataParallel(model)
            else:
                logger.info("The training will be carry out using 1 GPU: {}".format(torch.cuda.get_device_name(0)))
        else:
            logger.info("The training will be carry out using CPU")
            device = torch.device("cpu")
    else:
        print("The training will be carry out using 1 GPU:")
        print(torch.cuda.get_device_name(device))
        logger.info("The training will be carry out using 1 GPU: {}".format(torch.cuda.get_device_name(device)))


    # Moving the model to the given device
    model.to(device)

    # setting a flag for the early stop
    early_stop_count = 0
    best_epoch = 0
    best_metric_value = 1000 if best_metric == 'loss' else 0
    best_flag = False

    # Train history
    history = TrainHistory()

    # writer is used to generate the summary files to be loaded at tensorboard
    writer = SummaryWriter (os.path.join(save_folder, 'summaries'))

    # Let's iterate for `epoch` epochs or a tolerance.
    # It always start from epoch resume. If it's set, it starts from the last epoch the training phase was stopped,
    # otherwise, it starts from 0
    epoch = epoch_resume
    while epoch < epochs:

        # Updating epoch
        epoch += 1

        # Training and getting the metrics for one epoch
        train_metrics = _train_epoch(model, optimizer, loss_fn, train_data_loader, epoch, epochs, device)

        # After each epoch, we evaluate the model for the training and validation data
        val_metrics = metrics_for_eval(model, val_data_loader, device, loss_fn)

        # Checking the schedule if applicable
        if isinstance(schedule_lr, torch.optim.lr_scheduler.ReduceLROnPlateau):
            schedule_lr.step(val_metrics[best_metric])
        elif isinstance(schedule_lr, torch.optim.lr_scheduler.MultiStepLR):
            schedule_lr.step(epoch)

        # Getting the current LR
        current_LR = None
        for param_group in optimizer.param_groups:
            current_LR = param_group['lr']


        # Writing metrics for tensorboard
        writer.add_scalars('Loss', {'val-loss': val_metrics['loss'], 'train-loss': train_metrics['loss']}, epoch)
        writer.add_scalars('Dice', {'val-dice': val_metrics['dice'], 'train-loss': train_metrics['dice']}, epoch)
        writer.add_scalars('Jacc', {'val-jacc': val_metrics['jacc'], 'train-jacc': train_metrics['jacc']}, epoch)

        history.update(train_metrics['loss'], val_metrics['loss'],
                       train_metrics['dice'], val_metrics['dice'],
                       train_metrics['jacc'], val_metrics['jacc'])


        # Getting the metrics to print on logger
        train_print = "-- Loss: {:.3f}\n-- Dice: {:.3f}\n-- Jacc: {:.3f}".format(train_metrics["loss"],
                                                                              train_metrics["dice"],
                                                                              train_metrics["jacc"])

        val_print = "-- Loss: {:.3f}\n-- Dice: {:.3f}\n-- Jacc: {:.3f}".format(val_metrics["loss"],
                                                                            val_metrics["dice"],
                                                                            val_metrics["jacc"])

        early_stop_count += 1
        new_best_print = None
        # Defining the best metric for validation
        if best_metric == 'loss':
            if val_metrics[best_metric] <= best_metric_value:
                best_metric_value = val_metrics[best_metric]
                new_best_print = '-- New best {}: {:.3f}'.format(best_metric, best_metric_value)
                best_flag = True
                best_epoch = epoch
                early_stop_count = 0
        else:
            if val_metrics[best_metric] >= best_metric_value:
                best_metric_value = val_metrics[best_metric]
                new_best_print = '-- New best {}: {:.3f}'.format(best_metric, best_metric_value)
                best_flag = True
                best_epoch = epoch
                early_stop_count = 0

        # Check if it's the best model in order to save it
        if save_folder is not None:
            print ('- Saving the model...')
            save_model(model, save_folder, epoch, optimizer, loss_fn, best_flag, multi_gpu=m_gpu > 1)
        
        best_flag = False

        # Updating the logger
        msg = "Metrics for epoch {} out of {}\n".format(epoch, epochs)
        msg += "- Train\n"
        msg += train_print + "\n"
        msg += "\n- Validation\n"
        msg += val_print + "\n"
        msg += "\n- Training info"
        msg += "\n-- Early stopping counting: {} max to stop is {}".format(early_stop_count, epochs_early_stop)
        msg += "\n-- Current LR: {}".format(current_LR)
        if new_best_print is not None:
            msg += new_best_print
        msg += "\n-- Best {} so far: {:.3f} on epoch {}".format(best_metric, best_metric_value, best_epoch)


        # Checking the early stop
        if epochs_early_stop is not None:
            if early_stop_count >= epochs_early_stop:
                logger.info(msg)
                logger.info("The early stop trigger was activated. The validation {} " .format(best_metric) +
                            "{:.3f} did not improved for {} epochs.".format(best_metric_value,
                                                                            epochs_early_stop) +
                            "The training phase was stopped.")

                break

        # Checking the early stop
        if min_metric_early_stop is not None:
            stop = False
            if best_metric == 'loss':
                if min_metric_early_stop >= best_metric_value:
                    stop = True
            else:
                if min_metric_early_stop <= best_metric_value:
                    stop = True

            if stop:
                logger.info(msg)
                logger.info("The early stop trigger was activated. The validation {} ".format(best_metric) +
                            "{:.3f} achieved the defined threshold {:.3f}.".format(best_metric_value,
                                                                            min_metric_early_stop) +
                            "The training phase was stopped.")
                break

        # Sending all message to the logger
        logger.info(msg)

    if history_plot:
        history.save_plot(save_folder)

    history.save(save_folder)
    print('\n')

    writer.close()







