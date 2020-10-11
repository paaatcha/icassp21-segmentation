
import numpy as np
import os
import matplotlib.pyplot as plt


class AVGMetrics (object):
    """
        This is a simple class to control the AVG for a given value. It's used to control loss and accuracy for start
        and evaluate partition
    """
    def __init__(self):
        self.sum_value = 0
        self.avg = 0
        self.count = 0
        self.values = []

    def __call__(self):
        return self.avg

    def update(self, val):
        self.values.append(val)
        self.sum_value += val
        self.count += 1
        self.avg = self.sum_value / float(self.count)

    def print (self):
        print('\nsum_value: ', self.sum_value)
        print('count: ', self.count)
        print('avg: ', self.avg)


def dice_coeff (seg_pred, seg_true, threshold=0.5, smooth=1):
    '''
    dice = 2 * |X and Y| / |X| + |Y|
    '''
    seg_pred = seg_pred.view(-1)
    # seg_pred = (seg_pred > threshold).int()
    seg_true = seg_true.view(-1)
    # seg_true = (seg_true > threshold).int()
    intersection = (seg_pred * seg_true).sum()
    dice = (2. * intersection + smooth) / (seg_pred.sum() + seg_true.sum() + smooth)
    return dice


def jaccard_coeff (seg_pred, seg_true, threshold=0.5, smooth=1):
    """
    jac = |A and B| / |A or B|
    or
    jac = |A and B| / |A| + |B| - |A and B|
    """
    seg_pred = seg_pred.view(-1)
    # seg_pred = (seg_pred > threshold).int()
    seg_true = seg_true.view(-1)
    # seg_true = (seg_true > threshold).int()
    intersection = (seg_pred * seg_true).sum()

    total = (seg_pred + seg_true).sum()
    union = total - intersection
    jac = (intersection + smooth) / (union + smooth)
    return jac


class TrainHistory:

    def __init__(self):
        self.val_loss = list()
        self.val_dice = list()
        self.train_loss = list()
        self.train_dice = list()
        self.train_jac = list()
        self.val_jac = list()

    def update(self, loss_train, loss_val, dice_train, dice_val, jac_train, jac_val):

        self.train_loss.append(loss_train)
        self.val_loss.append(loss_val)
        self.train_dice.append(dice_train)
        self.val_dice.append(dice_val)
        self.train_jac.append(jac_train)
        self.val_jac.append(jac_val)

    def save(self, folder_path):

        path = os.path.join(folder_path, 'history')
        if not os.path.isdir(path):
            os.mkdir(path)

        print("Saving history CSVs in {}".format(path))

        np.savetxt(os.path.join(path, "train_loss.csv"), np.asarray(self.train_loss), fmt='%.3f', delimiter=',')
        np.savetxt(os.path.join(path, "val_loss.csv"), np.asarray(self.val_loss), fmt='%.3f', delimiter=',')

        np.savetxt(os.path.join(path, "train_dice.csv"), np.asarray(self.train_dice), fmt='%.3f', delimiter=',')
        np.savetxt(os.path.join(path, "val_dice.csv"), np.asarray(self.val_dice), fmt='%.3f', delimiter=',')

        np.savetxt(os.path.join(path, "train_jac.csv"), np.asarray(self.train_jac), fmt='%.3f', delimiter=',')
        np.savetxt(os.path.join(path, "val_jac.csv"), np.asarray(self.val_jac), fmt='%.3f', delimiter=',')

    def save_plot(self, folder_path):
        """
        This function saves a plot of the loss and accuracy history
        :param folder_path: a string with the base folder path
        """

        path = os.path.join(folder_path, 'history')
        if not os.path.isdir(path):
            os.mkdir(path)

        epochs = [i + 1 for i in range(len(self.train_loss))]

        print("Saving history plots in {}".format(path))

        plt.plot(epochs, self.train_loss, color='r', linestyle='solid')
        plt.plot(epochs, self.val_loss, color='b', linestyle='solid')
        plt.grid(color='black', linestyle='dotted', linewidth=0.7)
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss throughout epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "loss_history.png"), dpi=200)

        plt.figure()

        plt.plot(epochs, self.train_dice, color='r', linestyle='solid')
        plt.plot(epochs, self.val_dice, color='b', linestyle='solid')
        plt.grid(color='black', linestyle='dotted', linewidth=0.7)
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.xlabel("Epoch")
        plt.ylabel("Dice index")
        plt.title("Dice index throughout epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "dice_history.png"), dpi=200)

        plt.figure()

        plt.plot(epochs, self.train_jac, color='r', linestyle='solid')
        plt.plot(epochs, self.val_jac, color='b', linestyle='solid')
        plt.grid(color='black', linestyle='dotted', linewidth=0.7)
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.xlabel("Epoch")
        plt.ylabel("Jaccard index")
        plt.title("Jaccard index throughout epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(path, "jacc_history.png"), dpi=200)

        plt.figure()