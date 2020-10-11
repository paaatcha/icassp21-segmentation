import sys
sys.path.append("../")
import torch
from source.datasets import get_isic_dataloaders
from source.unet import UNet
from source.pipeline import fit_model, test_model, DiceBCELoss, JacLoss
import torch.nn as nn
from torch import optim
from sacred import Experiment
from sacred.observers import FileStorageObserver
import time
import os

# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():

    _unet_type = "bilinear" # "convtranspose"
    _model_name = "UNet_" + _unet_type
    _isic_path = "/home/patcha/Datasets/ISIC-Seg/splited"
    _save_folder = os.path.join("results", f"{_model_name}_{str(time.time()).replace('.','')}")
    _image_size = 256

    _batch_size = 7
    _aug_prob = 0.5

    _lr = 0.0001
    _epochs = 100
    _loss_type = "dice" # dice, bce, jac
    _epochs_early_stop = 15
    _training = True

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_model_name, _isic_path, _save_folder, _image_size, _batch_size, _aug_prob, _lr, _epochs, _loss_type,
          _epochs_early_stop, _training, _unet_type):

    train_dl, val_dl, test_dl = get_isic_dataloaders(_isic_path, _image_size, _batch_size, augmentation_prob=_aug_prob)

    model = UNet(3, 1, upsample_mode=_unet_type)

    if _loss_type == "dice":
        loss_func = DiceBCELoss()
    elif _loss_type == 'jac':
        loss_func = JacLoss()
    elif _loss_type == 'bce':
        loss_func = nn.BCEWithLogitsLoss()
    else:
        raise Exception(f"There is no loss type {_loss_type} available")

    if _training:
        opt = optim.SGD(model.parameters(), lr=_lr, weight_decay=1e-8, momentum=0.9)
        scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10)
    # scheduler_lr = optim.lr_scheduler.MultiStepLR(opt, [40, 60], gamma=0.1)

        fit_model(model, train_dl, val_dl, optimizer=opt, loss_fn=loss_func, epochs=_epochs,
                  epochs_early_stop=_epochs_early_stop, save_folder=_save_folder, initial_model=None, device=None,
                  schedule_lr=scheduler_lr, model_name=_model_name, history_plot=True, best_metric="loss")

    print("-" * 50)
    # Testing the test partition
    print("\n- Evaluating the validation partition...")
    test_model(model, test_dl, checkpoint_path=None, loss_fn=loss_func, device=None, save_path=_save_folder)





