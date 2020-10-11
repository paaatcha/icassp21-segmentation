import os
import argparse
import random
import shutil
from shutil import copyfile
from tqdm import tqdm
from glob import glob


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    img_list = os.listdir(config.origin_data_path)
    mask_list = os.listdir(config.origin_GT_path)

    img_list.sort()
    mask_list.sort()

    num_total = len(img_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for _ in tqdm(range(num_train), ncols=100, desc="Creating training set:", ascii=True):
        idx = Arange.pop()
        
        src = os.path.join(config.origin_data_path, img_list[idx])
        dst = os.path.join(config.train_path,img_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, mask_list[idx])
        dst = os.path.join(config.train_GT_path, mask_list[idx])
        copyfile(src, dst)
        

    for _ in tqdm(range(num_valid), ncols=100, desc="Creating validation set:", ascii=True):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, img_list[idx])
        dst = os.path.join(config.valid_path,img_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, mask_list[idx])
        dst = os.path.join(config.valid_GT_path, mask_list[idx])
        copyfile(src, dst)


    for _ in tqdm(range(num_test), ncols=100, desc="Creating test set:", ascii=True):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, img_list[idx])
        dst = os.path.join(config.test_path,img_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, mask_list[idx])
        dst = os.path.join(config.test_GT_path, mask_list[idx])
        copyfile(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    base_path = "/home/patcha/Datasets/Carvana"

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='/home/patcha/Datasets/Carvana/imgs')
    parser.add_argument('--origin_GT_path', type=str, default='/home/patcha/Datasets/Carvana/masks')

    parser.add_argument('--train_path', type=str, default=f'{base_path}/train/')
    parser.add_argument('--train_GT_path', type=str, default=f'{base_path}/train_GT/')
    parser.add_argument('--valid_path', type=str, default=f'{base_path}/valid/')
    parser.add_argument('--valid_GT_path', type=str, default=f'{base_path}/valid_GT/')
    parser.add_argument('--test_path', type=str, default=f'{base_path}/test/')
    parser.add_argument('--test_GT_path', type=str, default=f'{base_path}/test_GT/')


    config = parser.parse_args()
    print(config)
    main(config)