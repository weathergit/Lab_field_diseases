# -*- coding: utf-8 -*-
# @Time    : 2022/10/22 下午11:30
# @Author  : weather
# @Software: PyCharm
import os
from skimage.util import random_noise
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from skimage.io import imread, imsave
import numpy as np
from glob import glob


def show_class_number(root_dir):
    for fld in os.listdir(root_dir):
        path = os.path.join(root_dir, fld, '*')
        images = glob(path)
        print(fld, len(images))


def remove_augmentation(root_dir):
    for fld in os.listdir(root_dir):
        path = os.path.join(root_dir, fld, '*augmentation.jpg')
        images = glob(path)
        for fn in images:
            os.remove(fn)


def augmentation(fn, k):
    img = imread(fn)
    if img.shape != 3:
        img = img[:, :, :3]
    img30 = rotate(img, angle=30)
    img45 = rotate(img, angle=45)
    img60 = rotate(img, angle=60)
    img90 = rotate(img, angle=90)
    img_nosie = random_noise(img)
    imgud = np.flipud(img)
    imglr = np.fliplr(img)
    gama = adjust_gamma(img, gamma=0.5)
    gain = adjust_gamma(img, gain=0.5)
    augments = [img30, img_nosie, gama, gain, imglr, imgud, img45, img60, img90]
    names = ['img30', 'noise', 'imggama', 'imggain', 'imglr', 'imgud', 'img45', 'img60', 'img90']
    namelist = os.path.split(fn)
    outdir, basename = namelist[0], namelist[1][:-4]
    for ig, na in zip(augments[:k], names[:k]):
        if ig.dtype != np.uint8:
            ig = ig * 255
            ig = ig.astype('uint8')
        outname = outdir + '/' + na + '_' + basename + '_augmentation.jpg'
        imsave(outname, ig)


def do_aug(root_dir):
    for fld in os.listdir(root_dir):
        print(fld)
        path = os.path.join(root_dir, fld, '*')
        images = glob(path)
        n = int(1000/len(images))+1
        if n >= 2:
            k = n-1
            for fn in images:
                try:
                    augmentation(fn, k)
                except:
                    print(fn)


if __name__ == '__main__':

    image_dir = "./original_datasets/aug_split_datasets/lab/train/"
    show_class_number(image_dir)
    do_aug(image_dir)
    show_class_number(image_dir)
    #
    # remove_dir = './original_datasets/aug_split_datasets/lab/train/T'
    # remove_augmentation(remove_dir)
    # show_class_number(image_dir)
