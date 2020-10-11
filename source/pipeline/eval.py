#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com
"""

import torch
import torch.nn as nn
import pandas as pd
from source.utils import AVGMetrics, dice_coeff, jaccard_coeff
from .checkpoints import load_model
from tqdm import tqdm
import os
from torchvision.utils import save_image


import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def metrics_for_eval (model, data_loader, device, loss_fn):

    # setting the model to evaluation mode
    model.eval()

    print ("\nEvaluating...")
    with tqdm(total=len(data_loader), ascii=True, ncols=150) as t:

        # Setting require_grad=False in order to dimiss the gradient computation in the graph
        with torch.no_grad():

            # Variables to store the avg metrics
            loss_avg = AVGMetrics()
            jacc_avg = AVGMetrics()
            dice_avg = AVGMetrics()

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

                # Updating tqdm
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()), jacc='{:05.3f}'.format(jacc_avg()),
                              dice='{:05.3f}'.format(dice_avg()))
                t.update()

    return {"loss": loss_avg(), "jacc": jacc_avg(), "dice": dice_avg()}


def test_model (model, data_loader, checkpoint_path=None, loss_fn=None, device=None, save_path=None, save_masks=False):

    # setting the model to evaluation mode
    model.eval()

    if checkpoint_path is not None:
        model = load_model(checkpoint_path, model)

    if device is None:
        # Setting the device
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(torch.cuda.current_device()))
        else:
            device = torch.device("cpu")

    # Moving the model to the given device
    model.to(device)

    if loss_fn is None:
        loss_fn = nn.BCELoss()

    print("Testing...")
    with tqdm(total=len(data_loader), ascii=True, ncols=100) as t:

        # Setting require_grad=False in order to dimiss the gradient computation in the graph
        with torch.no_grad():

            # Variables to store the avg metrics
            loss_avg = AVGMetrics()
            jacc_avg = AVGMetrics()
            dice_avg = AVGMetrics()

            for data in data_loader:

                imgs_batch, mask_batch = data[0].to(device), data[1].to(device)
                seg_pred = model(imgs_batch)

                # if not save_masks:
                #     # print(seg_pred.shape)
                #     x = seg_pred[0].cpu()
                #
                #     x = x.numpy()
                #     plt.imshow(x[0])
                #     plt.show()
                #
                #     x = seg_pred[1].cpu()
                #
                #     x = x.numpy()
                #     plt.imshow(x[0])
                #     plt.show()
                #
                #     x = seg_pred[2].cpu()
                #
                #     x = x.numpy()
                #     plt.imshow(x[0])
                #     plt.show()
                #
                #     exit()
                #
                #     trans = transforms.ToPILImage(mode='RGB')
                #     x = trans(x)
                #     x.show()
                #     x.save("../../x.jpg")
                #     print(x)
                #     exit()
                #     # plt.imshow()
                #     # plt.show()
                #
                #     exit()
                #
                #     print(x.shape)

                # Computing loss function
                loss = loss_fn(seg_pred, mask_batch)
                jacc = jaccard_coeff(torch.sigmoid(seg_pred), mask_batch)
                dice = dice_coeff(torch.sigmoid(seg_pred), mask_batch)

                # Getting the avg metrics
                loss_avg.update(loss.item())
                jacc_avg.update(jacc.item())
                dice_avg.update(dice.item())

                # Updating tqdm
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()), jacc='{:05.3f}'.format(jacc_avg()),
                              dice='{:05.3f}'.format(dice_avg()))
                t.update()

    out_dict = {"loss": loss_avg(), "jacc": jacc_avg(), "dice": dice_avg()}

    if save_path is not None:
        df = pd.DataFrame.from_dict(out_dict.items())
        df.to_csv(os.path.join(save_path, 'test_metrics.csv'), index=False)

    return out_dict

