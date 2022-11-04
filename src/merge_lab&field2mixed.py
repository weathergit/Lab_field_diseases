# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午4:26
# @Author  : weather
# @Software: PyCharm


import os
import glob
import shutil


def getfiles(root_dir):
    criterion = os.path.join(root_dir, '*', '*', '*')
    images = glob.glob(criterion)
    return images


def replace_folder(imglist, src='field', target='mixed'):
    new_imglist = []
    for img in imglist:
        new = img.replace(src, target)
        new_imglist.append(new)
    return new_imglist


def recursive_copy(srclist, dstlist):
    for src, dst in zip(srclist, dstlist):
        dst_folder = os.path.split(dst)[0]
        print(dst_folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        shutil.copy(src, dst)


if __name__ == '__main__':
    field_imglist = getfiles('../data/original_datasets/aug_split_datasets/field')
    lab_imglist = getfiles('../data/original_datasets/aug_split_datasets/lab/')
    field_imglist_new = replace_folder(field_imglist, src='field')
    lab_imglist_new = replace_folder(lab_imglist, src='lab')
    recursive_copy(field_imglist, field_imglist_new)
    recursive_copy(lab_imglist, lab_imglist_new)
