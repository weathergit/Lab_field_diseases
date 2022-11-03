# -*- coding: utf-8 -*-
# @Time    : 2022/10/25 下午11:28
# @Author  : weather
# @Software: PyCharm

import os
import glob
import shutil
import numpy as np
from pprint import pprint


def get_images_number(src='./field/'):
    targets = ['train', 'valid', 'test']
    dis_num = {}
    for target in targets:
        fld = os.path.join(src, target)
        for dis in os.listdir(fld):
            criterion = os.path.join(fld, dis, '*')
            images = glob.glob(criterion)
            dis_num[target+' '+dis] = len(images)
    return dis_num


def copy_field():
    srd_floder = glob.glob('./field/[!o]*/*')
    for src in srd_floder:
        for i in range(1, 11):
            dst_folder = './partial/part{}'.format(str(i).zfill(2))
            dst = src.replace('./field', dst_folder)
            shutil.copytree(src, dst)


def get_lab_disease_images(folder='./lab/'):
    targets = ['train', 'valid', 'test']
    results = {}
    for target in targets:
        path = os.path.join(folder, target)
        sublist = os.listdir(path)
        for sub in sublist:
            criterion = os.path.join(path, sub, '*')
            images = glob.glob(criterion)
            results[target+'/'+sub] = images
    return results


def main():
    # copy_field()
    lab_images = get_lab_disease_images()
    for key, value in lab_images.items():
        image_number = len(value)
        for i in range(1, 11):
            out_path = './partial/part{}'.format(str(i).zfill(2))
            choice = int(image_number * i/10)
            np.random.seed(42)
            select_image = np.random.choice(value, choice, replace=False)
            for img in select_image:
                dst = img.replace('./lab', out_path)
                shutil.copy(img, dst)


if __name__ == "__main__":
    main()