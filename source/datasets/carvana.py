import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import os
from glob import glob


class CarvanaDataset(data.Dataset):

    def __init__(self, imgs_dir, masks_dir, mode, size=None, prob_aug=0.5, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.prob_aug = prob_aug
        self.mode = mode
        self.size = size

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]

        print(f"-- {len(self.ids)} images to {mode}...")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(os.path.join(self.masks_dir, idx + self.mask_suffix + '.*'))
        img_file = glob(os.path.join(self.imgs_dir, idx + '.*'))

        # print(os.path.join(self.masks_dir, idx + self.mask_suffix + '.*'))
        # print(idx, mask_file, img_file)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        if self.mode == 'train' and np.random.rand() <= self.prob_aug:
            _p = 0.5

            # Center crop
            if np.random.rand() <= _p:
                _aux = 1 - (np.random.rand() * 0.35)
                img = TF.center_crop(img, int(_aux * max(img.size)))
                mask = TF.center_crop(mask, int(_aux * max(mask.size)))

            # Brightness
            if np.random.rand() <= _p:
                _aux = 1 - (np.random.rand() * 0.4)
                img = TF.adjust_brightness(img, _aux)

            # Hue
            if np.random.rand() <= _p:
                _aux = np.random.rand() * 0.2
                img = TF.adjust_hue(img, _aux)

            # Contrast
            if np.random.rand() <= _p:
                _aux = 1 - (np.random.rand() * 0.5)
                img = TF.adjust_contrast(img, _aux)

            # Hflip
            if np.random.rand() <= _p:
                img = TF.hflip (img)
                mask = TF.hflip(mask)

            # Vflip
            if np.random.rand() <= _p:
                img = TF.vflip(img)
                mask = TF.vflip(mask)


        general_trans = T.Compose([T.Resize(self.size), T.ToTensor()])
        img = general_trans(img)
        mask = general_trans(mask)

        return img, mask


def get_carvana_dataloaders (base_path, image_size, batch_size, num_workers=16, augmentation_prob=0.5):

    print("-" * 50)
    print("- Loading ISIC dataset..")

    dataset_train = CarvanaDataset(os.path.join(base_path, "train"),
                                   os.path.join(base_path, "train_GT"),
                                   "train",
                                   (image_size, image_size),
                                   prob_aug=augmentation_prob,
                                   mask_suffix='_mask')

    dataset_val = CarvanaDataset(os.path.join(base_path, "valid"),
                                 os.path.join(base_path, "valid_GT"),
                                 "valid",
                                 (image_size, image_size),
                                 prob_aug=augmentation_prob,
                                 mask_suffix='_mask')

    dataset_test = CarvanaDataset(os.path.join(base_path, "test"),
                                  os.path.join(base_path, "test_GT"),
                                  "test",
                                  (image_size, image_size),
                                  prob_aug=augmentation_prob,
                                  mask_suffix='_mask')

    data_loader_train = data.DataLoader(dataset=dataset_train,
								        batch_size=batch_size,
								        shuffle=True,
								        num_workers=num_workers)

    data_loader_valid = data.DataLoader(dataset=dataset_val,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=num_workers)

    data_loader_test = data.DataLoader(dataset=dataset_test,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers)

    print("-" * 50)

    return data_loader_train, data_loader_valid, data_loader_test


# base_path = "/home/patcha/Datasets/Carvana"
# image_size = 256
# dataset_train = CarvanaDataset(os.path.join(base_path, "train"),
#                                os.path.join(base_path, "train_GT"),
#                                "train",
#                                (image_size, image_size),
#                                prob_aug=1,
#                                mask_suffix='_mask')
#
# for d in dataset_train:
#     pass