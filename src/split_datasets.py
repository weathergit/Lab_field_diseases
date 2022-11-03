# -*- coding: utf-8 -*-
# @Time    : 2022/10/22 下午9:34
# @Author  : weather
# @Software: PyCharm


from sklearn.model_selection import train_test_split
import os
import glob
import shutil


def movefile(images_list, dst, sub, target='train'):
    dst_folder = os.path.join(dst, target, sub)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for img in images_list:
        basename = os.path.basename(img)
        dst_path = os.path.join(dst_folder, basename)
        shutil.copy(img, dst_path)


def split(src, dst):
    for sub in os.listdir(src):
        criterion = os.path.join(src, sub, '*')
        images = glob.glob(criterion)
        train_data, valid_test = train_test_split(images, test_size=0.4, shuffle=True, random_state=42)
        valid_data, test_data = train_test_split(valid_test, test_size=0.5, shuffle=True, random_state=42)
        movefile(train_data, dst, sub, target='train')
        print(sub+' complete train datasets move!')
        movefile(valid_data, dst, sub, target='valid')
        print(sub+' complete valid datasets move!')
        movefile(test_data, dst, sub, target='test')
        print(sub+' complete test datasets move!')


if __name__ == "__main__":
    src = './original_datasets/merged_datasets/lab/'
    dst = './original_datasets/split_datasets/lab/'
    split(src, dst)